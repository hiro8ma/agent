package callguard_test

import (
	"context"
	"iter"
	"strings"
	"sync/atomic"
	"testing"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/genai"

	"github.com/hiro8ma/agent/go/internal/adk/callguard"
)

// looping は何を受けても同じツールを同じ引数で呼び、cancelAt 回目で context を切る。
type looping struct {
	calls    *atomic.Int32
	cancelAt int32
	cancel   context.CancelFunc
}

func (looping) Name() string { return "scripted" }

func (m looping) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	if m.calls.Add(1) == m.cancelAt {
		m.cancel()
	}
	part := &genai.Part{FunctionCall: &genai.FunctionCall{Name: "search", Args: map[string]any{"query": "在庫"}}}
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: &genai.Content{Role: "model", Parts: []*genai.Part{part}}}, nil)
	}
}

type result struct {
	modelCalls int32
	toolRuns   int32
	lastText   string
}

func run(t *testing.T, plugins ...*plugin.Plugin) result {
	t.Helper()
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	var calls, runs atomic.Int32
	search, err := functiontool.New(functiontool.Config{Name: "search", Description: "在庫を検索する"},
		func(agent.Context, struct {
			Query string `json:"query"`
		},
		) (map[string]any, error) {
			runs.Add(1)
			return map[string]any{"status": "no_result"}, nil
		})
	if err != nil {
		t.Fatal(err)
	}
	a, err := llmagent.New(llmagent.Config{
		Name: "loop", Model: looping{calls: &calls, cancelAt: 1000, cancel: cancel}, Tools: []tool.Tool{search},
	})
	if err != nil {
		t.Fatal(err)
	}
	r, err := runner.New(runner.Config{
		AppName: "ops", Agent: a, SessionService: session.InMemoryService(), AutoCreateSession: true,
		PluginConfig: runner.PluginConfig{Plugins: plugins},
	})
	if err != nil {
		t.Fatal(err)
	}
	var last string
	msg := genai.NewContentFromText("在庫を調べて", genai.RoleUser)
	for ev, err := range r.Run(ctx, "u", "s", msg, agent.RunConfig{}) {
		if err != nil {
			break
		}
		if ev.Content != nil && len(ev.Content.Parts) > 0 && ev.Content.Parts[0].Text != "" {
			last = ev.Content.Parts[0].Text
		}
	}
	return result{modelCalls: calls.Load(), toolRuns: runs.Load(), lastText: last}
}

func guard(t *testing.T, l callguard.Limits) *plugin.Plugin {
	t.Helper()
	p, err := callguard.New(l)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func TestWithoutGuardLoopsUntilContextIsCanceled(t *testing.T) {
	t.Parallel()
	got := run(t)
	if got.modelCalls != 1000 {
		t.Errorf("モデルの呼び出し = %d, want 1000（context を切るまで止まらない）", got.modelCalls)
	}
}

func TestGuardStopsLoops(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		limits         callguard.Limits
		wantModelCalls int32
		wantToolRuns   int32
		wantReason     string
	}{
		"同じ呼び出しが 3 回を超えたら 4 回目を実行せずに打ち切る": {
			limits:         callguard.Limits{MaxSameToolCalls: 3},
			wantModelCalls: 4, wantToolRuns: 3, wantReason: "同じ引数で 3 回",
		},
		"モデルの呼び出しが 5 回を超えたら 6 回目を呼ばずに打ち切る": {
			limits:         callguard.Limits{MaxLLMCalls: 5},
			wantModelCalls: 5, wantToolRuns: 5, wantReason: "5 回を超えた",
		},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			got := run(t, guard(t, tc.limits))
			if got.modelCalls != tc.wantModelCalls || got.toolRuns != tc.wantToolRuns {
				t.Errorf("モデル %d 回 / ツール %d 回, want %d / %d", got.modelCalls, got.toolRuns, tc.wantModelCalls, tc.wantToolRuns)
			}
			if !strings.Contains(got.lastText, tc.wantReason) {
				t.Errorf("最後の応答 = %q", got.lastText)
			}
		})
	}
}

func TestSamePluginCountsEachInvocationSeparately(t *testing.T) {
	t.Parallel()
	p := guard(t, callguard.Limits{MaxSameToolCalls: 3})
	first, second := run(t, p), run(t, p)
	if first.toolRuns != 3 || second.toolRuns != 3 {
		t.Errorf("2 回目の実行に 1 回目の回数が混ざっている: %d / %d", first.toolRuns, second.toolRuns)
	}
}
