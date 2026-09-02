package chapter01

import (
	"context"
	"fmt"
	"iter"
	"slices"
	"strings"
	"testing"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/genai"
)

// scriptedModel は台本どおりに応答するモデル。
//
// API キー無しで実行ループを流すために使う。モデルの賢さは測れないが、
// 「ツールが呼ばれ、結果が State に残り、応答に統合される」までの
// 配線は測れる。
type scriptedModel struct {
	turns []*genai.Content
	calls int
	seen  []string // モデルが受け取ったツール名
}

func (m *scriptedModel) Name() string { return "scripted" }

func (m *scriptedModel) GenerateContent(_ context.Context, req *model.LLMRequest,
	_ bool) iter.Seq2[*model.LLMResponse, error] {

	for name := range req.Tools {
		m.seen = append(m.seen, name)
	}
	i := m.calls
	m.calls++
	return func(yield func(*model.LLMResponse, error) bool) {
		if i >= len(m.turns) {
			yield(&model.LLMResponse{
				Content:      &genai.Content{Role: "model", Parts: []*genai.Part{{Text: "台本の終わり"}}},
				TurnComplete: true,
			}, nil)
			return
		}
		yield(&model.LLMResponse{Content: m.turns[i], TurnComplete: true}, nil)
	}
}

func call(name string, args map[string]any) *genai.Content {
	return &genai.Content{Role: "model", Parts: []*genai.Part{
		{FunctionCall: &genai.FunctionCall{Name: name, Args: args}},
	}}
}

func text(s string) *genai.Content {
	return &genai.Content{Role: "model", Parts: []*genai.Part{{Text: s}}}
}

// 観測 → 理解 → 計画 → 行動 → 評価 のループが 1 往復で回るかを見る。
//
// 台本は「天気を引く → 観光を引く → 統合して答える」の 3 手。
// 途中でツールが呼ばれなければ、統合された応答も出ない。
func TestExecutionLoopCallsBothToolsAndIntegrates(t *testing.T) {
	m := &scriptedModel{turns: []*genai.Content{
		call("get_weather", map[string]any{"city": "東京"}),
		call("get_sightseeing", map[string]any{"city": "東京"}),
		text("東京は晴れ、気温 28 度です。屋外の浅草寺がおすすめです。"),
	}}

	a, log, err := NewWithModel(m)
	if err != nil {
		t.Fatalf("build agent: %v", err)
	}
	svc := session.InMemoryService()
	r, err := runner.New(runner.Config{
		AppName: "ch01_loop", Agent: a, SessionService: svc, AutoCreateSession: true,
	})
	if err != nil {
		t.Fatalf("runner: %v", err)
	}

	var toolResults []string
	var payloads []string
	var final string
	for ev, err := range r.Run(context.Background(), "u1", "s1",
		&genai.Content{Role: "user", Parts: []*genai.Part{{Text: "東京に旅行に行きたいんだけど"}}},
		agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("run: %v", err)
		}
		if ev.Content == nil {
			continue
		}
		for _, p := range ev.Content.Parts {
			if p.FunctionResponse != nil {
				toolResults = append(toolResults, p.FunctionResponse.Name)
				payloads = append(payloads, fmt.Sprint(p.FunctionResponse.Response))
			}
			if p.Text != "" {
				final = p.Text
			}
		}
	}

	// 行動: 2 つのツールがこの順で実行されたか
	if got := strings.Join(toolResults, ","); got != "get_weather,get_sightseeing" {
		t.Errorf("ツールの実行順 = %q（get_weather,get_sightseeing のはず）", got)
	}

	// 理解: モデルに 2 つのツールが見えていたか。
	// 部分一致で見ると、名前を変えた対照が通ってしまう。
	for _, want := range []string{"get_weather", "get_sightseeing"} {
		if !slices.Contains(m.seen, want) {
			t.Errorf("モデルに %s が渡っていない（渡ったのは %v）", want, m.seen)
		}
	}

	// 行動の中身: ツールの戻り値に実データが入っているか。
	// 未登録のツールを呼んでも FunctionResponse は名前付きで返るため、
	// 名前だけ見ていては呼べたことにならない。
	joined := strings.Join(payloads, " ")
	for _, want := range []string{"晴れ", "浅草寺"} {
		if !strings.Contains(joined, want) {
			t.Errorf("ツールの戻り値に %q が無い: %s", want, joined)
		}
	}

	// 台本の最終応答が届いたか。中身は台本なので、これは経路の確認にとどまる。
	if final == "" {
		t.Error("最終応答が空")
	}

	// 記憶: 直近の都市が残ったか
	res, err := svc.Get(context.Background(),
		&session.GetRequest{AppName: "ch01_loop", UserID: "u1", SessionID: "s1"})
	if err != nil {
		t.Fatalf("session get: %v", err)
	}
	v, err := res.Session.State().Get(LastCityKey)
	if err != nil {
		t.Fatalf("%s が残っていない", LastCityKey)
	}
	if v != "tokyo" {
		t.Errorf("%s = %v（tokyo のはず）", LastCityKey, v)
	}

	// ガードレールが通過を数えているか
	if len(log.Passed()) == 0 {
		t.Error("ガードレールが 1 度も通過を数えていない")
	}
}

// 都市名が落ちた呼び出しがガードレールで止まるかを見る。
func TestMissingCityIsBlockedByGuardrail(t *testing.T) {
	m := &scriptedModel{turns: []*genai.Content{
		call("get_weather", map[string]any{}),
		text("どの都市について知りたいですか。"),
	}}

	a, log, err := NewWithModel(m)
	if err != nil {
		t.Fatalf("build agent: %v", err)
	}
	r, err := runner.New(runner.Config{
		AppName: "ch01_block", Agent: a,
		SessionService: session.InMemoryService(), AutoCreateSession: true,
	})
	if err != nil {
		t.Fatalf("runner: %v", err)
	}
	for _, err := range r.Run(context.Background(), "u1", "s1",
		&genai.Content{Role: "user", Parts: []*genai.Part{{Text: "天気を教えて"}}}, agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("run: %v", err)
		}
	}
	if len(log.Blocked()) == 0 {
		t.Error("city が無い呼び出しが止まっていない")
	}
}
