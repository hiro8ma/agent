package killswitch_test

import (
	"context"
	"errors"
	"iter"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
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

	"github.com/hiro8ma/agent/go/internal/adk/killswitch"
)

// scripted はツールを 1 回呼び、結果を受けたら答える。
type scripted struct{ calls *atomic.Int32 }

func (scripted) Name() string { return "scripted" }

func (m scripted) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	m.calls.Add(1)
	part := &genai.Part{FunctionCall: &genai.FunctionCall{Name: "delete_record", Args: map[string]any{"id": "r1"}}}
	if last := req.Contents[len(req.Contents)-1]; last.Parts[0].FunctionResponse != nil {
		part = &genai.Part{Text: "終わりました"}
	}
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: &genai.Content{Role: "model", Parts: []*genai.Part{part}}}, nil)
	}
}

type result struct {
	modelCalls int32
	toolRuns   int32
	lastText   string
}

func run(t *testing.T, sw *killswitch.Switch) result {
	t.Helper()
	var calls, runs atomic.Int32
	del, err := functiontool.New(functiontool.Config{Name: "delete_record", Description: "記録を消す"},
		func(agent.Context, struct {
			ID string `json:"id"`
		},
		) (map[string]any, error) {
			runs.Add(1)
			return map[string]any{"deleted": true}, nil
		})
	if err != nil {
		t.Fatal(err)
	}
	a, err := llmagent.New(llmagent.Config{Name: "ops_agent", Model: scripted{calls: &calls}, Tools: []tool.Tool{del}})
	if err != nil {
		t.Fatal(err)
	}
	p, err := sw.Plugin()
	if err != nil {
		t.Fatal(err)
	}
	r, err := runner.New(runner.Config{
		AppName: "ops", Agent: a, SessionService: session.InMemoryService(), AutoCreateSession: true,
		PluginConfig: runner.PluginConfig{Plugins: []*plugin.Plugin{p}},
	})
	if err != nil {
		t.Fatal(err)
	}
	var last string
	for ev, err := range r.Run(t.Context(), "u", "s", genai.NewContentFromText("消して", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			t.Fatal(err)
		}
		if ev.Content != nil && ev.Content.Parts[0].Text != "" {
			last = ev.Content.Parts[0].Text
		}
	}
	return result{modelCalls: calls.Load(), toolRuns: runs.Load(), lastText: last}
}

func TestScopes(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		state          killswitch.State
		wantModelCalls int32
		wantToolRuns   int32
	}{
		"止めていなければモデルもツールも動く":      {wantModelCalls: 2, wantToolRuns: 1},
		"全体を止めるとモデルを呼ばない":         {state: killswitch.State{Global: true}, wantModelCalls: 0},
		"エージェントを止めるとモデルを呼ばない":     {state: killswitch.State{Agents: []string{"ops_agent"}}, wantModelCalls: 0},
		"ほかのエージェントを止めても動く":        {state: killswitch.State{Agents: []string{"other"}}, wantModelCalls: 2, wantToolRuns: 1},
		"ツールを止めるとモデルは動くがツールは動かない": {state: killswitch.State{Tools: []string{"delete_record"}}, wantModelCalls: 2, wantToolRuns: 0},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			store := &killswitch.Memory{}
			store.Set(tc.state)
			got := run(t, killswitch.New(store, time.Minute))
			if got.modelCalls != tc.wantModelCalls || got.toolRuns != tc.wantToolRuns {
				t.Errorf("モデル %d 回 / ツール %d 回, want %d / %d", got.modelCalls, got.toolRuns, tc.wantModelCalls, tc.wantToolRuns)
			}
		})
	}
}

func TestOneStoreStopsEveryInstanceWithinTTL(t *testing.T) {
	t.Parallel()
	store := &killswitch.Memory{}
	now := time.Unix(0, 0)
	clock := func() time.Time { return now }
	a := killswitch.New(store, 10*time.Second).WithClock(clock)
	b := killswitch.New(store, 10*time.Second).WithClock(clock)
	ctx := t.Context()
	if a.Reason(ctx, "ops_agent", "") != "" || b.Reason(ctx, "ops_agent", "") != "" {
		t.Fatal("止める前から止まっている")
	}

	store.Set(killswitch.State{Global: true})
	now = now.Add(5 * time.Second)
	if a.Reason(ctx, "ops_agent", "") != "" {
		t.Error("TTL の間は手元の状態を使うはず")
	}
	now = now.Add(6 * time.Second)
	if a.Reason(ctx, "ops_agent", "") == "" || b.Reason(ctx, "ops_agent", "") == "" {
		t.Error("TTL を過ぎても、両方のインスタンスが止まっていない")
	}

	store.Set(killswitch.State{})
	now = now.Add(11 * time.Second)
	if a.Reason(ctx, "ops_agent", "") != "" {
		t.Error("解除した後も止まったまま")
	}
}

type broken struct{}

func (broken) Load(context.Context) (killswitch.State, error) {
	return killswitch.State{}, errors.New("保存先に届かない")
}

func TestFailsClosedWhenStoreIsUnreachable(t *testing.T) {
	t.Parallel()
	got := run(t, killswitch.New(broken{}, time.Minute))
	if got.modelCalls != 0 || got.toolRuns != 0 {
		t.Errorf("保存先が読めないのに動いた: モデル %d 回 / ツール %d 回", got.modelCalls, got.toolRuns)
	}
	if !strings.Contains(got.lastText, "確かめられない") {
		t.Errorf("応答 = %q", got.lastText)
	}
}

func TestFileStore(t *testing.T) {
	t.Parallel()
	path := filepath.Join(t.TempDir(), "killswitch.json")
	if err := os.WriteFile(path, []byte(`{"tools":["delete_record"]}`), 0o600); err != nil {
		t.Fatal(err)
	}
	got := run(t, killswitch.New(killswitch.File(path), time.Minute))
	if got.modelCalls != 2 || got.toolRuns != 0 {
		t.Errorf("モデル %d 回 / ツール %d 回", got.modelCalls, got.toolRuns)
	}
}
