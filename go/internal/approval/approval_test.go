package approval_test

import (
	"context"
	"errors"
	"iter"
	"sync"
	"testing"
	"testing/synctest"
	"time"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/genai"

	"github.com/hiro8ma/agent/go/internal/approval"
)

var rules = map[string]approval.Rule{
	"transfer_funds": func(args map[string]any) approval.Risk {
		amount, _ := args["amount"].(float64)
		switch {
		case amount >= 100_000_000:
			return approval.Forbidden
		case amount >= 1_000_000:
			return approval.High
		case amount >= 100_000:
			return approval.Medium
		}
		return approval.Low
	},
}

// scripted は呼び出しごとに 1 件ずつ送金し、台本が尽きたら答える。
type scripted struct {
	mu      sync.Mutex
	amounts []float64
	step    int
}

func (m *scripted) Name() string { return "scripted" }

func (m *scripted) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	m.mu.Lock()
	defer m.mu.Unlock()
	// 直前がツールの応答なら 1 往復が終わったので答える。
	last := req.Contents[len(req.Contents)-1]
	part := &genai.Part{Text: "処理しました"}
	if last.Role == "user" && last.Parts[0].FunctionResponse == nil && m.step < len(m.amounts) {
		part = &genai.Part{FunctionCall: &genai.FunctionCall{Name: "transfer_funds", Args: map[string]any{"amount": m.amounts[m.step], "to": "999-0001"}}}
		m.step++
	}
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: &genai.Content{Role: "model", Parts: []*genai.Part{part}}}, nil)
	}
}

type transferArgs struct {
	Amount float64 `json:"amount"`
	To     string  `json:"to"`
}

type env struct {
	svc      *approval.Service
	model    *scripted
	runner   *runner.Runner
	sessions session.Service
	executed *[]float64
}

func newEnv(t *testing.T) env {
	t.Helper()
	var executed []float64
	var mu sync.Mutex
	transfer, err := functiontool.New(functiontool.Config{Name: "transfer_funds", Description: "送金する"},
		func(_ agent.Context, in transferArgs) (map[string]any, error) {
			mu.Lock()
			defer mu.Unlock()
			executed = append(executed, in.Amount)
			return map[string]any{"status": "sent", "amount": in.Amount}, nil
		})
	if err != nil {
		t.Fatal(err)
	}
	svc := approval.NewService(rules, []string{"manager"}, time.Hour)
	m := &scripted{}
	a, err := llmagent.New(llmagent.Config{
		Name: "treasury", Model: m, Tools: []tool.Tool{transfer},
		BeforeToolCallbacks: []llmagent.BeforeToolCallback{approval.BeforeTool(svc)},
	})
	if err != nil {
		t.Fatal(err)
	}
	sessions := session.InMemoryService()
	r, err := runner.New(runner.Config{AppName: "treasury", Agent: a, SessionService: sessions, AutoCreateSession: true})
	if err != nil {
		t.Fatal(err)
	}
	return env{svc: svc, model: m, runner: r, sessions: sessions, executed: &executed}
}

// ask は 1 往復させ、ツールの応答を返す。
func (e env) ask(t *testing.T, user string, amounts ...float64) []map[string]any {
	t.Helper()
	e.model.mu.Lock()
	e.model.amounts, e.model.step = amounts, 0
	e.model.mu.Unlock()
	var out []map[string]any
	msg := &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "送金して"}}}
	for ev, err := range e.runner.Run(t.Context(), user, "s-"+user, msg, agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
		if ev.Content == nil {
			continue
		}
		for _, p := range ev.Content.Parts {
			if p.FunctionResponse != nil {
				out = append(out, p.FunctionResponse.Response)
			}
		}
	}
	return out
}

func TestRiskEscalation(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		amount     float64
		wantStatus string
		wantRisk   string
	}{
		"少額は自動で実行":         {amount: 50_000, wantStatus: "sent"},
		"中額は本人の確認を待つ":      {amount: 300_000, wantStatus: "pending_approval", wantRisk: "medium"},
		"高額は承認者の承認を待つ":     {amount: 1_500_000, wantStatus: "pending_approval", wantRisk: "high"},
		"上限を超える額は承認でも通さない": {amount: 200_000_000, wantStatus: "forbidden"},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			e := newEnv(t)
			got := e.ask(t, "alice", tc.amount)
			if len(got) != 1 || got[0]["status"] != tc.wantStatus {
				t.Fatalf("応答 = %v", got)
			}
			if tc.wantRisk != "" && got[0]["risk"] != tc.wantRisk {
				t.Errorf("risk = %v, want %v", got[0]["risk"], tc.wantRisk)
			}
		})
	}
}

func TestForgedStateDoesNotApprove(t *testing.T) {
	t.Parallel()
	e := newEnv(t)
	// REST の /run の stateDelta と同じく、クライアントが State に承認済みの印を書いた状態。
	_, err := e.sessions.Create(t.Context(), &session.CreateRequest{
		AppName: "treasury", UserID: "alice", SessionID: "s-alice",
		State: map[string]any{"approval:transfer_funds": "approved", "approved": true},
	})
	if err != nil {
		t.Fatal(err)
	}
	got := e.ask(t, "alice", 1_500_000)
	if got[0]["status"] != "pending_approval" || len(*e.executed) != 0 {
		t.Errorf("State の偽造で送金された: %v %v", got, *e.executed)
	}
}

func TestApprovalIsBoundToArgsAndUsedOnce(t *testing.T) {
	t.Parallel()
	e := newEnv(t)
	id := e.ask(t, "alice", 1_500_000)[0]["request_id"].(string)

	if err := e.svc.Decide("alice", id, true); !errors.Is(err, approval.ErrSelfApproval) {
		t.Fatalf("依頼者本人の承認 err = %v", err)
	}
	if err := e.svc.Decide("bob", id, true); !errors.Is(err, approval.ErrNotApprover) {
		t.Fatalf("承認者でない利用者の承認 err = %v", err)
	}
	if err := e.svc.Decide("manager", id, true); err != nil {
		t.Fatalf("Decide() error = %v", err)
	}

	// 承認した額と違う額は、新しい承認を待つ。
	if got := e.ask(t, "alice", 10_000_000); got[0]["status"] != "pending_approval" {
		t.Errorf("承認と違う額が通った: %v", got)
	}
	// 同じ額は 1 回だけ通る。
	if got := e.ask(t, "alice", 1_500_000); got[0]["status"] != "sent" {
		t.Errorf("承認した額が通らない: %v", got)
	}
	if got := e.ask(t, "alice", 1_500_000); got[0]["status"] != "pending_approval" {
		t.Errorf("承認を 2 回使えた: %v", got)
	}
	// 別の利用者は、他人の承認で実行できない。
	if got := e.ask(t, "mallory", 1_500_000); got[0]["status"] != "pending_approval" {
		t.Errorf("他人の承認で実行できた: %v", got)
	}
	if want := []float64{1_500_000}; len(*e.executed) != 1 || (*e.executed)[0] != want[0] {
		t.Errorf("実行された送金 = %v, want %v", *e.executed, want)
	}
}

func TestMediumRiskIsConfirmedByRequester(t *testing.T) {
	t.Parallel()
	e := newEnv(t)
	id := e.ask(t, "alice", 300_000)[0]["request_id"].(string)
	if err := e.svc.Decide("manager", id, true); !errors.Is(err, approval.ErrNotRequester) {
		t.Fatalf("中リスクを他人が確認 err = %v", err)
	}
	if err := e.svc.Decide("alice", id, true); err != nil {
		t.Fatalf("Decide() error = %v", err)
	}
	if got := e.ask(t, "alice", 300_000); got[0]["status"] != "sent" {
		t.Errorf("本人が確認した送金が通らない: %v", got)
	}
}

func TestApprovalExpires(t *testing.T) {
	t.Parallel()
	synctest.Test(t, func(t *testing.T) {
		svc := approval.NewService(rules, []string{"manager"}, time.Hour)
		args := map[string]any{"amount": 1_500_000.0, "to": "999-0001"}
		r, err := svc.Open("alice", "transfer_funds", args, approval.High)
		if err != nil {
			t.Fatal(err)
		}
		if err := svc.Decide("manager", r.ID, true); err != nil {
			t.Fatal(err)
		}

		// 仮想の時計で 1 時間進める。実際には待たない。
		time.Sleep(time.Hour)
		if ok, _ := svc.Consume("alice", "transfer_funds", args); ok {
			t.Error("期限の切れた承認で実行できた")
		}

		r2, _ := svc.Open("alice", "transfer_funds", args, approval.High)
		time.Sleep(time.Hour + time.Second)
		if err := svc.Decide("manager", r2.ID, true); !errors.Is(err, approval.ErrExpired) {
			t.Errorf("期限切れの依頼を承認できた: %v", err)
		}
	})
}

func TestDigestIgnoresKeyOrder(t *testing.T) {
	t.Parallel()
	a, _ := approval.Digest("f", map[string]any{"a": 1, "b": 2})
	b, _ := approval.Digest("f", map[string]any{"b": 2, "a": 1})
	c, _ := approval.Digest("g", map[string]any{"a": 1, "b": 2})
	if a != b || a == c {
		t.Errorf("digest a=%s b=%s c=%s", a, b, c)
	}
}
