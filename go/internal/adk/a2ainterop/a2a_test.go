package a2ainterop_test

import (
	"context"
	"encoding/json"
	"iter"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2acompat/a2av0"
	"github.com/a2aproject/a2a-go/v2/a2asrv"
	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	adka2a "google.golang.org/adk/v2/server/adka2a/v2"
	"google.golang.org/adk/v2/session"
	"google.golang.org/genai"

	"github.com/hiro8ma/agent/go/internal/adk/a2ainterop"
)

// echo は受け取った文に「受け付けました」と返す台本のモデル。
type echo struct{}

func (echo) Name() string { return "scripted" }

func (echo) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	text := req.Contents[len(req.Contents)-1].Parts[0].Text
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: &genai.Content{Role: "model", Parts: []*genai.Part{{Text: "受け付けました: " + text}}}}, nil)
	}
}

func testCard() a2a.AgentCard {
	return a2a.AgentCard{
		Name: "expense-agent", Description: "経費精算を扱う", Version: "1.0.0",
		DefaultInputModes: []string{"text/plain"}, DefaultOutputModes: []string{"text/plain"},
		Skills: []a2a.AgentSkill{{ID: "expense", Name: "経費", Description: "経費の照会", Tags: []string{"expense"}}},
	}
}

func echoAgent(t *testing.T) agent.Agent {
	t.Helper()
	a, err := llmagent.New(llmagent.Config{Name: "expense_agent", Model: echo{}, Instruction: "x"})
	if err != nil {
		t.Fatal(err)
	}
	return a
}

// newServer は ADK のエージェントを A2A v1.0 のサーバーとして立て、/v0 に v0.3 の互換の口を置く。
func newServer(t *testing.T) *httptest.Server {
	t.Helper()
	srv := httptest.NewUnstartedServer(nil)
	srv.Start()
	t.Cleanup(srv.Close)
	srv.Config.Handler = a2ainterop.NewHandler(echoAgent(t), testCard(), srv.URL)
	return srv
}

// newServerWithoutV0Interface は Agent Card にプロトコル版 0.3 の接続先を書かないサーバー。
func newServerWithoutV0Interface(t *testing.T) *httptest.Server {
	t.Helper()
	exec := adka2a.NewExecutor(adka2a.ExecutorConfig{RunnerConfig: runner.Config{
		AppName: "expense", Agent: echoAgent(t), SessionService: session.InMemoryService(),
	}})
	mux := http.NewServeMux()
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	card := testCard()
	card.SupportedInterfaces = []*a2a.AgentInterface{a2a.NewAgentInterface(srv.URL+"/", a2a.TransportProtocolJSONRPC)}
	mux.Handle("/", a2asrv.NewJSONRPCHandler(a2asrv.NewHandler(exec)))
	mux.Handle(a2asrv.WellKnownAgentCardPath, a2asrv.NewAgentCardHandler(a2av0.NewStaticAgentCardProducer(&card)))
	return srv
}

func post(t *testing.T, url, body string) map[string]any {
	t.Helper()
	req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, url, strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	var out map[string]any
	if err := json.NewDecoder(res.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	return out
}

func get(t *testing.T, url string) map[string]any {
	t.Helper()
	req, err := http.NewRequestWithContext(t.Context(), http.MethodGet, url, nil)
	if err != nil {
		t.Fatal(err)
	}
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	var out map[string]any
	if err := json.NewDecoder(res.Body).Decode(&out); err != nil {
		t.Fatalf("GET %s: status %d: %v", url, res.StatusCode, err)
	}
	return out
}

const (
	v1Send = `{"jsonrpc":"2.0","id":1,"method":"SendMessage","params":{"message":{"messageId":"m1","role":"ROLE_USER","parts":[{"text":"先月の経費"}]}}}`
	v0Send = `{"jsonrpc":"2.0","id":1,"method":"message/send","params":{"message":{"messageId":"m1","role":"user","kind":"message","parts":[{"kind":"text","text":"先月の経費"}]}}}`
)

func TestV1ClientTalksToV1Server(t *testing.T) {
	t.Parallel()
	srv := newServer(t)
	res := post(t, srv.URL+"/", v1Send)
	raw, _ := json.Marshal(res)
	if res["error"] != nil || !strings.Contains(string(raw), "受け付けました: 先月の経費") {
		t.Fatalf("応答 = %s", raw)
	}
	if !strings.Contains(string(raw), "TASK_STATE_COMPLETED") && !strings.Contains(string(raw), "ROLE_AGENT") {
		t.Errorf("v1.0 の値（TASK_STATE_* / ROLE_*）が無い: %s", raw)
	}
}

func TestV0ClientCannotTalkToV1ServerByDefault(t *testing.T) {
	t.Parallel()
	srv := newServer(t)
	res := post(t, srv.URL+"/", v0Send)
	e, ok := res["error"].(map[string]any)
	if !ok {
		t.Fatalf("v0.3 の message/send が v1.0 の口で通った: %v", res)
	}
	if code := e["code"].(float64); code != -32601 {
		t.Errorf("error.code = %v, want -32601（メソッドが無い）", code)
	}
}

func TestV0ClientTalksThroughCompatEndpoint(t *testing.T) {
	t.Parallel()
	srv := newServer(t)
	res := post(t, srv.URL+"/v0", v0Send)
	raw, _ := json.Marshal(res)
	if res["error"] != nil || !strings.Contains(string(raw), "受け付けました: 先月の経費") {
		t.Fatalf("応答 = %s", raw)
	}
	if strings.Contains(string(raw), "TASK_STATE_") || strings.Contains(string(raw), "ROLE_AGENT") {
		t.Errorf("v0.3 の口が v1.0 の値を返した: %s", raw)
	}
}

func TestCompatAgentCardServesBothVersions(t *testing.T) {
	t.Parallel()
	srv := newServer(t)
	card := get(t, srv.URL+a2asrv.WellKnownAgentCardPath)

	// v0.3 のクライアントは直下の url と protocolVersion を読み、v1.0 のクライアントは supportedInterfaces を読む。
	if card["url"] != srv.URL+"/v0" || card["protocolVersion"] != "0.3" {
		t.Errorf("v0.3 の項目 url=%v protocolVersion=%v", card["url"], card["protocolVersion"])
	}
	ifaces, _ := card["supportedInterfaces"].([]any)
	if len(ifaces) != 2 {
		t.Errorf("supportedInterfaces = %v", card["supportedInterfaces"])
	}
}

func TestCompatAgentCardFailsWithoutV0Interface(t *testing.T) {
	t.Parallel()
	srv := newServerWithoutV0Interface(t)
	req, err := http.NewRequestWithContext(t.Context(), http.MethodGet, srv.URL+a2asrv.WellKnownAgentCardPath, nil)
	if err != nil {
		t.Fatal(err)
	}
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	// プロトコル版 0.3 の接続先を書き忘れると、理由を返さずに 500 になる。
	if res.StatusCode != http.StatusInternalServerError {
		t.Errorf("status = %d, want 500", res.StatusCode)
	}
}
