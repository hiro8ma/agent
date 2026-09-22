package usagemeter_test

import (
	"context"
	"iter"
	"maps"
	"math"
	"slices"
	"testing"
	"time"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/genai"

	"github.com/hiro8ma/agent/go/internal/adk/usagemeter"
)

var (
	toolCallUsage = &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: 1000, CachedContentTokenCount: 400, CandidatesTokenCount: 100, ThoughtsTokenCount: 50}
	answerUsage   = &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: 1200, CachedContentTokenCount: 400, ToolUsePromptTokenCount: 30, CandidatesTokenCount: 200}
)

// expense は 1 回目にツールを呼び、2 回目に答える台本のモデル。ストリーミングでは途中の応答にも使用量を付ける。
type expense struct{ name string }

func (e expense) Name() string { return e.name }

func (expense) GenerateContent(_ context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	last := req.Contents[len(req.Contents)-1]
	final := &model.LLMResponse{
		Content:       &genai.Content{Role: "model", Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{Name: "count_expenses", Args: map[string]any{}}}}},
		UsageMetadata: toolCallUsage,
	}
	if last.Parts[0].FunctionResponse != nil {
		final = &model.LLMResponse{Content: genai.NewContentFromText("7月の経費は12件です", genai.RoleModel), UsageMetadata: answerUsage}
	}
	return func(yield func(*model.LLMResponse, error) bool) {
		if stream && final.Content.Parts[0].Text != "" {
			for _, chunk := range []string{"7月の", "経費は12件です"} {
				partial := &model.LLMResponse{Content: genai.NewContentFromText(chunk, genai.RoleModel), UsageMetadata: answerUsage, Partial: true}
				if !yield(partial, nil) {
					return
				}
			}
		}
		yield(final, nil)
	}
}

func run(t *testing.T, m *usagemeter.Meter, modelName string, mode agent.StreamingMode) {
	t.Helper()
	count, err := functiontool.New(functiontool.Config{Name: "count_expenses", Description: "経費を数える"},
		func(_ agent.Context, _ struct{}) (map[string]any, error) { return map[string]any{"count": 12}, nil })
	if err != nil {
		t.Fatal(err)
	}
	a, err := llmagent.New(llmagent.Config{Name: "expense_agent", Model: expense{name: modelName}, Tools: []tool.Tool{count}})
	if err != nil {
		t.Fatal(err)
	}
	p, err := m.Plugin()
	if err != nil {
		t.Fatal(err)
	}
	r, err := runner.New(runner.Config{
		AppName: "office", Agent: a, SessionService: session.InMemoryService(), AutoCreateSession: true,
		PluginConfig: runner.PluginConfig{Plugins: []*plugin.Plugin{p}},
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, err := range r.Run(t.Context(), "alice", "s1", genai.NewContentFromText("7月の経費は？", genai.RoleUser), agent.RunConfig{StreamingMode: mode}) {
		if err != nil {
			t.Fatal(err)
		}
	}
}

var tokyo = time.FixedZone("JST", 9*60*60)

func TestCountsEachCallOnce(t *testing.T) {
	t.Parallel()
	want := usagemeter.Usage{Calls: 2, Prompt: 2200, Cached: 800, ToolUsePrompt: 30, Output: 300, Thoughts: 50}
	testCases := map[string]struct{ mode agent.StreamingMode }{
		"ストリーミングなし": {mode: agent.StreamingModeNone},
		"ストリーミングでも途中の応答の使用量を重ねて数えない": {mode: agent.StreamingModeSSE},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			m := usagemeter.New(nil, 0, tokyo)
			run(t, m, "gemini-3.8-flash", tc.mode)
			got := m.Report().Usage
			if !maps.Equal(got, map[usagemeter.Key]usagemeter.Usage{{Agent: "expense_agent", Model: "gemini-3.8-flash"}: want}) {
				t.Errorf("usage = %+v, want %+v", got, want)
			}
		})
	}
}

func TestCostAndBudget(t *testing.T) {
	t.Parallel()
	prices := map[string]usagemeter.Price{"gemini-3.8-flash": {Input: 2, CachedInput: 0.5, Output: 8}}
	m := usagemeter.New(prices, 0.01, tokyo)
	run(t, m, "gemini-3.8-flash", agent.StreamingModeNone)
	run(t, m, "unlisted-model", agent.StreamingModeNone)
	r := m.Report()
	// キャッシュを除いた入力 1400 とツールの結果 30 を入力の単価、キャッシュ 800 をキャッシュの単価、出力 300 と思考 50 を出力の単価で数える。
	wantCost := (1430*2 + 800*0.5 + 350*8) / 1e6
	if math.Abs(r.Cost-wantCost) > 1e-12 || math.Abs(r.BudgetRatio-wantCost/0.01) > 1e-9 {
		t.Errorf("cost = %v ratio = %v, want %v / %v", r.Cost, r.BudgetRatio, wantCost, wantCost/0.01)
	}
	if !slices.Equal(r.Unpriced, []string{"unlisted-model"}) {
		t.Errorf("unpriced = %v", r.Unpriced)
	}
}

func TestResetsAtLocalMidnight(t *testing.T) {
	t.Parallel()
	now := time.Date(2026, 9, 22, 23, 59, 0, 0, tokyo)
	m := usagemeter.New(nil, 0, tokyo)
	m.SetClock(func() time.Time { return now })
	run(t, m, "gemini-3.8-flash", agent.StreamingModeNone)
	if r := m.Report(); r.Day != "2026-09-22" || len(r.Usage) != 1 {
		t.Fatalf("report = %+v", r)
	}
	// UTC では同じ日でも、日本時間で日付が変われば数え直す。
	now = time.Date(2026, 9, 22, 15, 1, 0, 0, time.UTC)
	if r := m.Report(); r.Day != "2026-09-23" || len(r.Usage) != 0 {
		t.Errorf("report = %+v", r)
	}
}
