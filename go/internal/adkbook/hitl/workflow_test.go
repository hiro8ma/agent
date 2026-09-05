package hitl

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"
	"google.golang.org/genai"
)

func askOver(limit float64) Policy {
	return Threshold("amount", limit, limit*100, "円")
}

// pending は中断の識別子。再開はこれを付けた function response で返す。
type pending struct {
	id, name, message string
}

// step は 1 往復流し、中断の有無とノードの出力を返す。
//
// ノードの出力は Content ではなく Event.Output に入る。
// Content だけ見ていると、出力が無いように見える。
func step(t *testing.T, r *runner.Runner, sid string, part *genai.Part) (*pending, []string) {
	t.Helper()
	msg := &genai.Content{Role: "user", Parts: []*genai.Part{part}}

	var p *pending
	var out []string
	for ev, err := range r.Run(context.Background(), "u1", sid, msg, agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("run: %v", err)
		}
		if ev.Output != nil {
			b, _ := json.Marshal(ev.Output)
			out = append(out, string(b))
		}
		if ev.Content == nil {
			continue
		}
		for _, part := range ev.Content.Parts {
			fc := part.FunctionCall
			if fc == nil || fc.Name != workflow.WorkflowInputFunctionCallName {
				continue
			}
			msg, _ := fc.Args["message"].(string)
			p = &pending{id: fc.ID, name: fc.Name, message: msg}
		}
	}
	return p, out
}

func userText(s string) *genai.Part { return &genai.Part{Text: s} }

// reply は console launcher と同じ形で返答を組む。
// 平文は {"payload": ...} に包まれる。
func reply(p *pending, answer string) *genai.Part {
	return &genai.Part{FunctionResponse: &genai.FunctionResponse{
		ID: p.id, Name: p.name,
		Response: map[string]any{"payload": answer},
	}}
}

// buildRequest は利用者の入力を Request に変える前段。
//
// 承認ノードの入力は上流ノードが作る。実運用でも人の生入力を
// そのまま渡すことはない。
func buildRequest(t *testing.T) workflow.Node {
	t.Helper()
	return workflow.NewEmittingFunctionNode[any, Request]("build",
		func(_ agent.Context, in any, _ func(*session.Event) error) (Request, error) {
			var req Request
			if s, ok := in.(string); ok {
				if err := json.Unmarshal([]byte(s), &req); err == nil {
					return req, nil
				}
			}
			return Request{Action: "refund", Args: map[string]any{"amount": 5000}}, nil
		},
		workflow.NodeConfig{},
	)
}

func newRunner(t *testing.T, node workflow.Node, name string) *runner.Runner {
	t.Helper()
	a, err := workflowagent.New(workflowagent.Config{
		Name:  name,
		Edges: workflow.Chain(workflow.Start, buildRequest(t), node),
	})
	if err != nil {
		t.Fatalf("workflow agent: %v", err)
	}
	r, err := runner.New(runner.Config{
		AppName: name, Agent: a,
		SessionService: session.InMemoryService(), AutoCreateSession: true,
	})
	if err != nil {
		t.Fatalf("runner: %v", err)
	}
	return r
}

// 承認が要る額で中断し、承認すると通ることを見る。
func TestApprovalNodeInterruptsThenResumes(t *testing.T) {
	r := newRunner(t, ApprovalNode("refund", askOver(1000)), "hitl_ok")

	pend, _ := step(t, r, "s1", userText(`{"action":"refund","args":{"amount":5000}}`))
	if pend == nil {
		t.Fatal("承認を求めていない")
	}
	if !strings.Contains(pend.message, "refund") {
		t.Errorf("確認文に対象が無い: %q", pend.message)
	}

	_, out := step(t, r, "s1", reply(pend, "はい"))
	joined := strings.Join(out, " ")
	if !strings.Contains(joined, "approved") {
		t.Errorf("再開後に approved が出ていない: %s", joined)
	}
}

// 承認しなければ rejected になることを見る。
func TestApprovalNodeRejects(t *testing.T) {
	r := newRunner(t, ApprovalNode("refund", askOver(1000)), "hitl_ng")

	pend, _ := step(t, r, "s1", userText(`{"action":"refund","args":{"amount":5000}}`))
	if pend == nil {
		t.Fatal("承認を求めていない")
	}
	_, out := step(t, r, "s1", reply(pend, "いいえ"))
	joined := strings.Join(out, " ")
	if !strings.Contains(joined, "rejected") {
		t.Errorf("却下されていない: %s", joined)
	}
}

// しきい値以下では人に聞かないことを見る。
//
// 何でも聞くなら止める仕組みは要らない。聞かない側が動いて初めて機構になる。
func TestApprovalNodeDoesNotAskBelowThreshold(t *testing.T) {
	r := newRunner(t, ApprovalNode("refund", askOver(1000)), "hitl_low")

	pend, out := step(t, r, "s1", userText(`{"action":"refund","args":{"amount":10}}`))
	if pend != nil {
		t.Error("しきい値以下なのに承認を求めた")
	}
	if joined := strings.Join(out, " "); !strings.Contains(joined, "allowed") {
		t.Errorf("allowed が出ていない: %s", joined)
	}
}

// RerunOnResume が &true であることを見る。
//
// 既定の nil は handoff で、人の返答は後続ノードの入力になる。
// ResumeOrRequestInput は再実行を前提にしているため、
// nil のままだとこのノードは返答を受け取れない。
func TestApprovalNodeRequiresRerunOnResume(t *testing.T) {
	cfg := ApprovalNode("refund", askOver(1000)).Config()
	if cfg.RerunOnResume == nil {
		t.Fatal("RerunOnResume が nil。既定は handoff なので返答を受け取れない")
	}
	if !*cfg.RerunOnResume {
		t.Error("RerunOnResume が false")
	}
}

// 対照。RerunOnResume を既定（nil）にすると返答を受け取れないことを見る。
//
// ApprovalNode と同じ本体を nil 設定で組む。nil は handoff なので
// エンジンはこのノードを再実行せず、返答は後続ノードの入力になる。
// ResumeOrRequestInput は ctx.ResumedInput() で再実行を検知するため、
// 再実行されなければ承認は永久に成立しない。
func TestDefaultRerunOnResumeNeverSeesTheReply(t *testing.T) {
	body := func(ctx agent.Context, in Request, emit func(*session.Event) error) (Result, error) {
		r, err := workflow.ResumeOrRequestInput(ctx, emit, session.RequestInput{
			InterruptID: "ctl-" + ctx.InvocationID(),
			Message:     "承認しますか",
		})
		if err != nil {
			return Result{}, err
		}
		if approved(r) {
			return Result{Status: "approved"}, nil
		}
		return Result{Status: "rejected"}, nil
	}

	for _, tc := range []struct {
		name string
		cfg  workflow.NodeConfig
		want string
	}{
		{"既定（nil = handoff）", workflow.NodeConfig{}, "approved が出てはいけない"},
		{"&true（re-entry）", workflow.NodeConfig{RerunOnResume: ptr(true)}, "approved が出るはず"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			node := workflow.NewEmittingFunctionNode[Request, Result]("ctl", body, tc.cfg)
			r := newRunner(t, node, "hitl_ctl_"+tc.name)

			pend, _ := step(t, r, "s1", userText(`{"action":"x","args":{}}`))
			if pend == nil {
				t.Fatal("中断していない")
			}
			_, out := step(t, r, "s1", reply(pend, "はい"))
			got := strings.Contains(strings.Join(out, " "), `"status":"approved"`)

			if tc.cfg.RerunOnResume == nil && got {
				t.Errorf("%s のに approved が出た: %v", tc.want, out)
			}
			if tc.cfg.RerunOnResume != nil && !got {
				t.Errorf("%s のに出ていない: %v", tc.want, out)
			}
		})
	}
}

func ptr[T any](v T) *T { return &v }
