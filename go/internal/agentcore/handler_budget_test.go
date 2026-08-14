package agentcore_test

import (
	"context"
	"iter"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"

	"connectrpc.com/connect"

	agentv1 "github.com/hiro8ma/agent/go/gen/agent/v1"
	"github.com/hiro8ma/agent/go/gen/agent/v1/agentv1connect"
	"github.com/hiro8ma/agent/go/internal/agentcore"
)

// stubAgent はモデルを呼ばずに固定の応答とトークン消費を返す。
// LLM に依存せず Connect ハンドラとトークン予算の挙動だけを検証するために使う。
type stubAgent struct {
	tokensPerCall int
}

func (stubAgent) Info() agentcore.AgentInfo {
	return agentcore.AgentInfo{ID: "stub", Description: "テスト用の固定応答エージェント"}
}

func (s stubAgent) Ask(_ context.Context, _ *agentcore.AskInput) iter.Seq2[*agentcore.AskChunk, *agentcore.AskOutput] {
	return func(yield func(*agentcore.AskChunk, *agentcore.AskOutput) bool) {
		if !yield(&agentcore.AskChunk{AnswerDelta: "ok"}, nil) {
			return
		}
		yield(nil, &agentcore.AskOutput{
			Answer:       "ok",
			FinishReason: "stop",
			Usage:        agentcore.TokenUsage{TotalTokens: s.tokensPerCall},
		})
	}
}

type stubSessions struct{}

func (stubSessions) Load(context.Context, string) ([]agentcore.Message, error) { return nil, nil }
func (stubSessions) Append(context.Context, string, ...agentcore.Message) error {
	return nil
}

type stubExecutor struct{}

func (stubExecutor) Execute(context.Context, string) (map[string]any, error) {
	return nil, agentcore.ErrPendingNotFound
}

func newTestServer(t *testing.T, limits agentcore.BudgetLimits, tokensPerCall int) agentv1connect.AgentServiceClient {
	t.Helper()

	registry := agentcore.NewRegistry(stubAgent{tokensPerCall: tokensPerCall})
	h := agentcore.NewHandler(registry, stubSessions{}, stubExecutor{}, slog.New(slog.DiscardHandler))
	if limits.Enabled() {
		h = h.WithBudget(agentcore.NewBudgetTracker(limits))
	}

	mux := http.NewServeMux()
	path, handler := agentv1connect.NewAgentServiceHandler(h)
	mux.Handle(path, handler)

	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)

	return agentv1connect.NewAgentServiceClient(srv.Client(), srv.URL)
}

func ask(t *testing.T, c agentv1connect.AgentServiceClient, sessionID string) error {
	t.Helper()

	stream, err := c.Ask(context.Background(), connect.NewRequest(&agentv1.AskRequest{
		AgentId:   "stub",
		SessionId: sessionID,
		Message:   "hello",
	}))
	if err != nil {
		return err
	}
	defer func() { _ = stream.Close() }()

	for stream.Receive() {
	}
	return stream.Err()
}

// 予算を設定しなければ何度でも通る。
func TestAskWithoutBudget(t *testing.T) {
	c := newTestServer(t, agentcore.BudgetLimits{}, 1000)

	for i := range 3 {
		if err := ask(t, c, "s1"); err != nil {
			t.Fatalf("%d 回目で失敗した: %v", i+1, err)
		}
	}
}

// セッション上限を超えた次の呼び出しが resource_exhausted で拒否される。
func TestAskSessionBudgetExceeded(t *testing.T) {
	c := newTestServer(t, agentcore.BudgetLimits{SessionTokens: 100}, 150)

	if err := ask(t, c, "s1"); err != nil {
		t.Fatalf("1 回目は通るはずが失敗した: %v", err)
	}

	err := ask(t, c, "s1")
	if err == nil {
		t.Fatal("2 回目は拒否されるはずが通った")
	}
	if got := connect.CodeOf(err); got != connect.CodeResourceExhausted {
		t.Fatalf("コードが違う: got %v, want %v", got, connect.CodeResourceExhausted)
	}
}

// セッション上限は他のセッションに影響しない。
func TestAskSessionBudgetIsPerSession(t *testing.T) {
	c := newTestServer(t, agentcore.BudgetLimits{SessionTokens: 100}, 150)

	if err := ask(t, c, "s1"); err != nil {
		t.Fatalf("s1 の 1 回目が失敗した: %v", err)
	}
	if err := ask(t, c, "s1"); err == nil {
		t.Fatal("s1 の 2 回目は拒否されるはず")
	}
	if err := ask(t, c, "s2"); err != nil {
		t.Fatalf("別セッション s2 は通るはずが失敗した: %v", err)
	}
}

// 全体上限はセッションをまたいで効く。
func TestAskTotalBudgetExceeded(t *testing.T) {
	c := newTestServer(t, agentcore.BudgetLimits{TotalTokens: 100}, 150)

	if err := ask(t, c, "s1"); err != nil {
		t.Fatalf("1 回目が失敗した: %v", err)
	}
	if err := ask(t, c, "s2"); err == nil {
		t.Fatal("別セッションでも全体上限で拒否されるはず")
	}
}
