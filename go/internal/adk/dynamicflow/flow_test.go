package dynamicflow

import (
	"context"
	"fmt"
	"iter"
	"strings"
	"sync"
	"testing"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"
	"google.golang.org/genai"
)

// counting は台本どおり返しつつ、呼ばれた回数を数えるモデル。
//
// 完了済みの子ノードが再実行されるかを、この回数で判定する。
type counting struct {
	mu    sync.Mutex
	name  string
	lines []string
	calls int
}

func (m *counting) Name() string { return m.name }

func (m *counting) Calls() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.calls
}

func (m *counting) GenerateContent(_ context.Context, _ *model.LLMRequest,
	_ bool) iter.Seq2[*model.LLMResponse, error] {

	m.mu.Lock()
	i := m.calls
	m.calls++
	m.mu.Unlock()

	text := "（台本の終わり）"
	if i < len(m.lines) {
		text = m.lines[i]
	}
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{
			Content:      &genai.Content{Role: "model", Parts: []*genai.Part{{Text: text}}},
			TurnComplete: true,
		}, nil)
	}
}

func node(t *testing.T, name string, m model.LLM) workflow.Node {
	t.Helper()
	a, err := llmagent.New(llmagent.Config{Name: name, Model: m, Instruction: "台本"})
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	n, err := workflow.NewAgentNode(a, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("%s node: %v", name, err)
	}
	return n
}

func runFlow(t *testing.T, app string, draftM, reviewM model.LLM) []string {
	t.Helper()
	loop := New(node(t, "draft", draftM), node(t, "review", reviewM))
	a, err := workflowagent.New(workflowagent.Config{
		Name:  app,
		Edges: workflow.Chain(workflow.Start, loop),
	})
	if err != nil {
		t.Fatalf("workflow agent: %v", err)
	}
	r, err := runner.New(runner.Config{
		AppName: app, Agent: a,
		SessionService: session.InMemoryService(), AutoCreateSession: true,
	})
	if err != nil {
		t.Fatalf("runner: %v", err)
	}

	var out []string
	msg := &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "短い紹介文を書いて"}}}
	for ev, err := range r.Run(context.Background(), "u1", "s1", msg, agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("run: %v", err)
		}
		if ev.Output != nil {
			out = append(out, fmt.Sprint(ev.Output))
		}
	}
	return out
}

// 1 周目で承認されたら、そこで止まることを見る。
func TestApprovesOnFirstRound(t *testing.T) {
	draft := &counting{name: "d", lines: []string{"初稿"}}
	review := &counting{name: "r", lines: []string{Approved}}

	out := runFlow(t, "dyn_ok", draft, review)
	joined := strings.Join(out, " ")

	if !strings.Contains(joined, "rounds:1") && !strings.Contains(joined, "1") {
		t.Errorf("周回数が出ていない: %s", joined)
	}
	if !strings.Contains(joined, "true") {
		t.Errorf("承認で終わっていない: %s", joined)
	}
	if draft.Calls() != 1 {
		t.Errorf("下書きが %d 回。承認されたので 1 回のはず", draft.Calls())
	}
}

// 承認されなければ上限で打ち切ることを見る。
//
// 上限が効かなければ、評価者が承認しない限り止まらない。
func TestStopsAtMaxRounds(t *testing.T) {
	draft := &counting{name: "d", lines: []string{"1", "2", "3", "4"}}
	review := &counting{name: "r", lines: []string{"直して", "まだ直して", "もっと直して", "直して"}}

	out := runFlow(t, "dyn_max", draft, review)
	joined := strings.Join(out, " ")

	if !strings.Contains(joined, "false") {
		t.Errorf("未承認で終わっていない: %s", joined)
	}
	if got := review.Calls(); got != MaxRounds {
		t.Errorf("評価が %d 回。上限 %d 回のはず", got, MaxRounds)
	}
	// 初稿 1 回 + 各周の修正 3 回
	if got := draft.Calls(); got != MaxRounds+1 {
		t.Errorf("下書きが %d 回。初稿 1 + 修正 %d のはず", got, MaxRounds)
	}
}

// 上限をコードで持っていることを見る。
//
// LLM に「承認されるまで」を任せると止まる保証が無い。
func TestMaxRoundsIsBoundedInCode(t *testing.T) {
	if MaxRounds <= 0 || MaxRounds > 10 {
		t.Errorf("MaxRounds = %d。上限として意味のある値でない", MaxRounds)
	}
}

// 再開時に完了済みの子ノードが再実行されるかを数える。
//
// 教材は rerun_on_resume=True で「完了済みの子ノードを無駄に
// 再実行せず」と書いている。RerunOnResume の意味は
// 「中断したノードを最初から再実行する」なので、
// 子の呼び出し回数が増えるかどうかで真偽が決まる。
func TestResumeReexecutesCompletedChildren(t *testing.T) {
	draft := &counting{name: "d", lines: []string{"初稿", "初稿"}}
	review := &counting{name: "r", lines: []string{"よい", "よい"}}

	loop := NewWithApproval(node(t, "draft", draft), node(t, "review", review))
	a, err := workflowagent.New(workflowagent.Config{
		Name: "dyn_hitl", Edges: workflow.Chain(workflow.Start, loop),
	})
	if err != nil {
		t.Fatalf("workflow agent: %v", err)
	}
	r, err := runner.New(runner.Config{
		AppName: "dyn_hitl", Agent: a,
		SessionService: session.InMemoryService(), AutoCreateSession: true,
	})
	if err != nil {
		t.Fatalf("runner: %v", err)
	}

	step := func(part *genai.Part) (id, name string, out []string) {
		msg := &genai.Content{Role: "user", Parts: []*genai.Part{part}}
		for ev, err := range r.Run(context.Background(), "u1", "s1", msg, agent.RunConfig{}) {
			if err != nil {
				t.Fatalf("run: %v", err)
			}
			if ev.Output != nil {
				out = append(out, fmt.Sprint(ev.Output))
			}
			if ev.Content == nil {
				continue
			}
			for _, p := range ev.Content.Parts {
				if fc := p.FunctionCall; fc != nil && fc.Name == workflow.WorkflowInputFunctionCallName {
					id, name = fc.ID, fc.Name
				}
			}
		}
		return id, name, out
	}

	id, name, _ := step(&genai.Part{Text: "紹介文を書いて"})
	if id == "" {
		t.Fatal("中断していない")
	}
	beforeDraft, beforeReview := draft.Calls(), review.Calls()
	t.Logf("中断時点   下書き %d 回 / 評価 %d 回", beforeDraft, beforeReview)

	_, _, out := step(&genai.Part{FunctionResponse: &genai.FunctionResponse{
		ID: id, Name: name, Response: map[string]any{"payload": "はい"},
	}})
	afterDraft, afterReview := draft.Calls(), review.Calls()
	t.Logf("再開後     下書き %d 回 / 評価 %d 回", afterDraft, afterReview)
	t.Logf("最終出力   %v", out)

	if strings.Join(out, " ") == "" {
		t.Error("再開後に出力が無い")
	}

	// 実測した挙動を固定する。教材は再実行されないと書いているが逆だった。
	// 将来のバージョンで変わったら、この検査が落ちて気づける。
	if afterDraft <= beforeDraft {
		t.Errorf("下書きが再実行されていない（%d → %d）。挙動が変わった可能性がある",
			beforeDraft, afterDraft)
	}
	if afterReview <= beforeReview {
		t.Errorf("評価が再実行されていない（%d → %d）。挙動が変わった可能性がある",
			beforeReview, afterReview)
	}
}

// 重い処理を別ノードへ出すと再実行されないことを見る。
//
// 再入の対照。中断するノードの手前で完了させておけば、
// 再開しても走り直さない。
func TestHandoffDoesNotReexecuteCompletedNodes(t *testing.T) {
	draft := &counting{name: "d", lines: []string{"初稿", "初稿"}}
	review := &counting{name: "r", lines: []string{"よい", "よい"}}

	a, err := workflowagent.New(workflowagent.Config{
		Name:  "dyn_handoff",
		Edges: NewHandoff(node(t, "draft", draft), node(t, "review", review)),
	})
	if err != nil {
		t.Fatalf("workflow agent: %v", err)
	}
	r, err := runner.New(runner.Config{
		AppName: "dyn_handoff", Agent: a,
		SessionService: session.InMemoryService(), AutoCreateSession: true,
	})
	if err != nil {
		t.Fatalf("runner: %v", err)
	}

	step := func(part *genai.Part) (id, name string, out []string) {
		msg := &genai.Content{Role: "user", Parts: []*genai.Part{part}}
		for ev, err := range r.Run(context.Background(), "u1", "s1", msg, agent.RunConfig{}) {
			if err != nil {
				t.Fatalf("run: %v", err)
			}
			if ev.Output != nil {
				out = append(out, fmt.Sprint(ev.Output))
			}
			if ev.Content == nil {
				continue
			}
			for _, p := range ev.Content.Parts {
				if fc := p.FunctionCall; fc != nil && fc.Name == workflow.WorkflowInputFunctionCallName {
					id, name = fc.ID, fc.Name
				}
			}
		}
		return id, name, out
	}

	id, name, _ := step(&genai.Part{Text: "紹介文を書いて"})
	if id == "" {
		t.Fatal("中断していない")
	}
	beforeDraft, beforeReview := draft.Calls(), review.Calls()
	t.Logf("中断時点   下書き %d 回 / 評価 %d 回", beforeDraft, beforeReview)

	_, _, out := step(&genai.Part{FunctionResponse: &genai.FunctionResponse{
		ID: id, Name: name, Response: map[string]any{"payload": "はい"},
	}})
	t.Logf("再開後     下書き %d 回 / 評価 %d 回", draft.Calls(), review.Calls())
	t.Logf("最終出力   %v", out)

	if draft.Calls() != beforeDraft {
		t.Errorf("下書きが再実行された（%d → %d）", beforeDraft, draft.Calls())
	}
	if review.Calls() != beforeReview {
		t.Errorf("評価が再実行された（%d → %d）", beforeReview, review.Calls())
	}
	if !strings.Contains(strings.Join(out, " "), "true") {
		t.Errorf("承認が届いていない: %v", out)
	}
}
