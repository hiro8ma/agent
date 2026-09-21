package adkeval

import (
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

type parity struct {
	Bigram []struct {
		Candidate string  `json:"candidate"`
		Reference string  `json:"reference"`
		Score     float64 `json:"score"`
	} `json:"bigram"`
	Trajectory []struct {
		Name      string    `json:"name"`
		MatchType string    `json:"match_type"`
		Actual    []ToolUse `json:"actual"`
		Expected  []ToolUse `json:"expected"`
		Score     float64   `json:"score"`
	} `json:"trajectory"`
}

func loadParity(t *testing.T) parity {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "parity.json"))
	if err != nil {
		t.Fatalf("ReadFile() error = %v", err)
	}
	var p parity
	if err := json.Unmarshal(raw, &p); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	return p
}

// Python の ADK と samples/evaluation/ja_response_match.py が出した値と、同じ入力で同じ値になる。
func TestScoresMatchPythonADK(t *testing.T) {
	t.Parallel()
	p := loadParity(t)
	modes := map[string]MatchType{"EXACT": Exact, "IN_ORDER": InOrder, "ANY_ORDER": AnyOrder}

	for _, c := range p.Bigram {
		t.Run("bigram/"+c.Candidate, func(t *testing.T) {
			t.Parallel()
			if got := BigramF1(c.Candidate, c.Reference); math.Abs(got-c.Score) > 1e-9 {
				t.Errorf("BigramF1(%q, %q) = %v, Python は %v", c.Candidate, c.Reference, got, c.Score)
			}
		})
	}
	for _, c := range p.Trajectory {
		t.Run("trajectory/"+c.Name+"/"+c.MatchType, func(t *testing.T) {
			t.Parallel()
			actual := []Invocation{{IntermediateData: IntermediateData{ToolUses: c.Actual}}}
			expected := []Invocation{{IntermediateData: IntermediateData{ToolUses: c.Expected}}}
			if got := TrajectoryScore(actual, expected, modes[c.MatchType]); got != c.Score {
				t.Errorf("TrajectoryScore = %v, Python の ADK は %v", got, c.Score)
			}
		})
	}
}

// fakeADK は ADK の REST を真似る。発話ごとに台本のイベントを返し、受けた発話を記録する。
type fakeADK struct {
	mu       sync.Mutex
	script   map[string][]Event
	messages []string
	sessions int
}

func (f *fakeADK) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	defer f.mu.Unlock()
	switch {
	case strings.HasSuffix(r.URL.Path, "/sessions"):
		f.sessions++
		_ = json.NewEncoder(w).Encode(map[string]string{"id": fmt.Sprintf("s%d", f.sessions)})
	case strings.HasSuffix(r.URL.Path, "/run"):
		var req struct {
			NewMessage Content `json:"newMessage"`
			SessionID  string  `json:"sessionId"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		text := req.NewMessage.Text()
		f.messages = append(f.messages, req.SessionID+":"+text)
		events, ok := f.script[text]
		if !ok {
			http.Error(w, "no script", http.StatusInternalServerError)
			return
		}
		_ = json.NewEncoder(w).Encode(events)
	default:
		http.NotFound(w, r)
	}
}

func call(name string, args map[string]any) Event {
	return Event{Author: "weather_agent", Content: &Content{Role: "model", Parts: []Part{{FunctionCall: &ToolUse{Name: name, Args: args}}}}}
}

func say(author, text string) Event {
	return Event{Author: author, Content: &Content{Role: "model", Parts: []Part{{Text: text}}}}
}

func userTurn(text string, response string, calls ...ToolUse) Invocation {
	inv := Invocation{
		UserContent:      Content{Role: "user", Parts: []Part{{Text: text}}},
		IntermediateData: IntermediateData{ToolUses: calls},
	}
	if response != "" {
		inv.FinalResponse = &Content{Role: "model", Parts: []Part{{Text: response}}}
	}
	return inv
}

func TestCompareScoresEachTargetAndGates(t *testing.T) {
	t.Parallel()
	good := &fakeADK{script: map[string][]Event{
		"東京の天気は？": {call("get_weather", map[string]any{"city": "tokyo"}), {Author: "weather_agent", Partial: true, Content: &Content{Parts: []Part{{Text: "東京"}}}}, say("weather_agent", "東京は晴れ、25度です。")},
		"映画は？":    {say("weather_agent", "天気以外は答えられません。")},
	}}
	bad := &fakeADK{script: map[string][]Event{
		"東京の天気は？": {call("get_weather", map[string]any{"city": "東京"}), say("weather_agent", "東京は晴れ、25度です。")},
		"映画は？":    {call("get_weather", map[string]any{"city": "tokyo"}), say("weather_agent", "映画ならこちら。")},
	}}
	set := &EvalSet{EvalCases: []EvalCase{
		{EvalID: "tokyo", Conversation: []Invocation{userTurn("東京の天気は？", "東京は晴れ、25度です。", ToolUse{Name: "get_weather", Args: map[string]any{"city": "tokyo"}})}},
		{EvalID: "out_of_scope", Conversation: []Invocation{userTurn("映画は？", "")}},
	}}
	goodSrv, badSrv := httptest.NewTestServer(t, good), httptest.NewTestServer(t, bad)
	targets := []Target{
		{Name: "good", Agent: &RESTAgent{BaseURL: goodSrv.URL, AppName: "weather_agent", HTTP: goodSrv.Client()}},
		{Name: "bad", Agent: &RESTAgent{BaseURL: badSrv.URL, AppName: "weather_agent", HTTP: badSrv.Client()}},
	}

	results := Compare(t.Context(), set, targets, Options{Match: Exact})
	th := Thresholds{Trajectory: 1, Response: 0.5}

	got := map[string]bool{}
	for _, r := range results {
		got[r.Target+"/"+r.EvalID] = r.Passed(th)
	}
	want := map[string]bool{
		"good/tokyo": true, "good/out_of_scope": true,
		"bad/tokyo": false, "bad/out_of_scope": false,
	}
	for k, w := range want {
		if got[k] != w {
			t.Errorf("%s の合否 = %v, want %v", k, got[k], w)
		}
	}

	if AllPassed(results, th) {
		t.Error("失敗したケースがあるのに AllPassed が true")
	}
}

func TestInferKeepsOneSessionAcrossTurns(t *testing.T) {
	t.Parallel()
	f := &fakeADK{script: map[string][]Event{
		"京都に 2 泊したい":    {call("search_hotels", map[string]any{"location": "Kyoto", "nights": 2}), say("a", "候補です")},
		"1 泊 15000 円以内": {call("search_hotels", map[string]any{"location": "Kyoto", "nights": 2, "max_price": 15000}), say("a", "3 件です")},
	}}
	srv := httptest.NewTestServer(t, f)
	agent := &RESTAgent{BaseURL: srv.URL, AppName: "trip", HTTP: srv.Client()}
	c := EvalCase{EvalID: "kyoto", Conversation: []Invocation{
		userTurn("京都に 2 泊したい", ""), userTurn("1 泊 15000 円以内", ""),
	}}

	actual, err := Infer(t.Context(), agent, "u1", c)
	if err != nil {
		t.Fatalf("Infer() error = %v", err)
	}
	if f.messages[0][:3] != f.messages[1][:3] {
		t.Errorf("ターンごとにセッションが変わった: %v", f.messages)
	}
	if got := actual[1].IntermediateData.ToolUses[0].Args["max_price"]; got != float64(15000) {
		t.Errorf("2 ターン目の max_price = %v", got)
	}
}

func TestFallbackResponseIsCountedAsFailureNotScored(t *testing.T) {
	t.Parallel()
	f := &fakeADK{script: map[string][]Event{
		"東京の天気は？": {{
			Author: "weather_agent", ErrorCode: "MODEL_ERROR_FALLBACK", ErrorMessage: "モデルの呼び出しに失敗し、決まった文で答えた",
			Content: &Content{Role: "model", Parts: []Part{{Text: "いま天気を取得できません。"}}},
		}},
	}}
	srv := httptest.NewTestServer(t, f)
	set := &EvalSet{EvalCases: []EvalCase{{EvalID: "tokyo", Conversation: []Invocation{userTurn("東京の天気は？", "")}}}}
	results := Compare(t.Context(), set, []Target{{Name: "go", Agent: &RESTAgent{BaseURL: srv.URL, AppName: "a", HTTP: srv.Client()}}}, Options{})

	// 呼ばないことを期待するケースでも、失敗の置き換えは合格にしない。
	if results[0].Passed(Thresholds{}) || !strings.Contains(results[0].Error, "MODEL_ERROR_FALLBACK") {
		t.Errorf("結果 = %+v", results[0])
	}
}

func TestInferReportsServerErrors(t *testing.T) {
	t.Parallel()
	srv := httptest.NewTestServer(t, &fakeADK{script: map[string][]Event{}})
	agent := &RESTAgent{BaseURL: srv.URL, AppName: "a", HTTP: srv.Client()}
	_, err := Infer(t.Context(), agent, "u1", EvalCase{EvalID: "x", Conversation: []Invocation{userTurn("台本に無い", "")}})
	if err == nil || !strings.Contains(err.Error(), "x の 1 ターン目") {
		t.Errorf("err = %v", err)
	}
}

func TestLoadRejectsEmptyTurns(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		body    string
		wantErr string
	}{
		"ケースが無い":       {body: `{"eval_set_id":"x","eval_cases":[]}`, wantErr: "ケースが無い"},
		"発話の無いターン":     {body: `{"eval_set_id":"x","eval_cases":[{"eval_id":"c","conversation":[{"user_content":{"parts":[]}}]}]}`, wantErr: "利用者の発話が無い"},
		"Python の形を読む": {body: `{"eval_set_id":"x","eval_cases":[{"eval_id":"c","conversation":[{"user_content":{"role":"user","parts":[{"text":"q"}]},"intermediate_data":{"tool_uses":[{"name":"f","args":{"a":1}}],"tool_responses":[]}}]}]}`},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			path := filepath.Join(t.TempDir(), "set.evalset.json")
			if err := os.WriteFile(path, []byte(tc.body), 0o600); err != nil {
				t.Fatal(err)
			}
			_, err := Load(path)
			if tc.wantErr == "" {
				if err != nil {
					t.Fatalf("Load() error = %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Errorf("Load() error = %v, want %q", err, tc.wantErr)
			}
		})
	}
}
