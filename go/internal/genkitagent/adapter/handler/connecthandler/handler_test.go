package connecthandler_test

import (
	"context"
	"errors"
	"iter"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"

	"connectrpc.com/connect"

	agentv1 "github.com/hiro8ma/agent/go/gen/agent/v1"
	"github.com/hiro8ma/agent/go/gen/agent/v1/agentv1connect"
	"github.com/hiro8ma/agent/go/internal/conversation/repository"
	convusecase "github.com/hiro8ma/agent/go/internal/conversation/usecase"
	"github.com/hiro8ma/agent/go/internal/genkitagent/adapter/handler/connecthandler"
	"github.com/hiro8ma/agent/go/internal/genkitagent/adapter/infrastructure/conversationclient"
	"github.com/hiro8ma/agent/go/internal/genkitagent/adapter/infrastructure/inmemory"
	"github.com/hiro8ma/agent/go/internal/genkitagent/domain/model"
	sessionrepo "github.com/hiro8ma/agent/go/internal/genkitagent/domain/repository"
	"github.com/hiro8ma/agent/go/internal/genkitagent/domain/service"
	"github.com/hiro8ma/agent/go/internal/genkitagent/usecase"
	"github.com/hiro8ma/agent/go/internal/lib/libbudget"
	"github.com/hiro8ma/agent/go/internal/lib/libconnect"
)

// stubAgent はモデルを呼ばずに固定の応答とトークン消費を返す。
type stubAgent struct {
	tokensPerCall int
}

func (stubAgent) Info() model.AgentInfo {
	return model.AgentInfo{ID: "stub", Description: "テスト用の固定応答エージェント"}
}

func (s stubAgent) Chat(_ context.Context, _ *model.ChatInput) iter.Seq2[*model.ChatChunk, *model.ChatOutput] {
	return func(yield func(*model.ChatChunk, *model.ChatOutput) bool) {
		if !yield(&model.ChatChunk{AnswerDelta: "ok"}, nil) {
			return
		}
		yield(nil, &model.ChatOutput{
			Answer:       "ok",
			FinishReason: "stop",
			Usage:        model.TokenUsage{TotalTokens: s.tokensPerCall},
		})
	}
}

type noPendingGate struct{}

func (noPendingGate) Authorize(context.Context, string, map[string]any) (*model.ActionDecision, error) {
	return &model.ActionDecision{Outcome: model.ActionDeny}, nil
}

func (noPendingGate) Take(context.Context, string) (*model.ApprovedAction, error) {
	return nil, model.ErrPendingNotFound
}

// failingAppend は履歴の読み出しはできるが、追記に失敗する保存先。
type failingAppend struct{ sessionrepo.Session }

func (failingAppend) CreateMessages(context.Context, string, []model.Message) error {
	return errors.New("disk full")
}

func newAgentServer(t *testing.T, sessions sessionrepo.Session, opts ...usecase.Option) agentv1connect.AgentServiceClient {
	t.Helper()
	return newAgentServerWith(t, stubAgent{tokensPerCall: 1}, sessions, opts...)
}

func newAgentServerWith(t *testing.T, a service.Agent, sessions sessionrepo.Session, opts ...usecase.Option) agentv1connect.AgentServiceClient {
	t.Helper()
	uc := usecase.NewAgentService(service.NewRegistry(a), sessions, noPendingGate{}, inmemory.NewOrders(), slog.New(slog.DiscardHandler), opts...)
	mux := http.NewServeMux()
	mux.Handle(connecthandler.Route(connecthandler.New(uc), libconnect.HeaderAuthenticator))
	srv := httptest.NewTestServer(t, mux)
	return agentv1connect.NewAgentServiceClient(srv.Client(), srv.URL)
}

func localSessions(t *testing.T) *conversationclient.Local {
	t.Helper()
	repo, err := repository.NewSQLite("file:" + filepath.Join(t.TempDir(), "conversation.db"))
	if err != nil {
		t.Fatalf("NewSQLite() error = %v", err)
	}
	return conversationclient.NewLocal(convusecase.New(repo))
}

type chatResult struct {
	deltas []string
	result *agentv1.ChatResult
}

func chatAs(ctx context.Context, c agentv1connect.AgentServiceClient, user, sessionID string) (chatResult, error) {
	req := connect.NewRequest(&agentv1.ChatRequest{AgentId: "stub", SessionId: sessionID, Message: "hello"})
	if user != "" {
		req.Header().Set(libconnect.UserHeader, user)
	}
	stream, err := c.Chat(ctx, req)
	if err != nil {
		return chatResult{}, err
	}
	defer func() { _ = stream.Close() }()
	var out chatResult
	for stream.Receive() {
		switch ev := stream.Msg().GetEvent().(type) {
		case *agentv1.ChatResponse_AnswerDelta:
			out.deltas = append(out.deltas, ev.AnswerDelta)
		case *agentv1.ChatResponse_Result:
			out.result = ev.Result
		}
	}
	return out, stream.Err()
}

func TestChatCreatesSessionWhenIDIsEmpty(t *testing.T) {
	t.Parallel()
	c := newAgentServer(t, localSessions(t))

	got, err := chatAs(t.Context(), c, "alice", "")
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}
	if got.result.GetSessionId() == "" {
		t.Fatal("新しいセッションの ID が返らない")
	}
	if !got.result.GetHistorySaved() {
		t.Error("履歴に残ったのに history_saved が false")
	}
	if len(got.deltas) != 1 || got.deltas[0] != "ok" {
		t.Errorf("差分 = %v, want [ok]", got.deltas)
	}

	if _, err := chatAs(t.Context(), c, "alice", got.result.GetSessionId()); err != nil {
		t.Fatalf("2 回目の Chat() error = %v", err)
	}
}

func TestChatRequiresSessionIDWhenStoreCannotCreate(t *testing.T) {
	t.Parallel()
	c := newAgentServer(t, inmemory.NewSessions())

	_, err := chatAs(t.Context(), c, "alice", "")
	if connect.CodeOf(err) != connect.CodeInvalidArgument {
		t.Errorf("code = %v, want InvalidArgument", connect.CodeOf(err))
	}
}

func TestChatRefusesOtherUsersSession(t *testing.T) {
	t.Parallel()
	c := newAgentServer(t, localSessions(t))

	first, err := chatAs(t.Context(), c, "alice", "")
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}

	// 他人のセッションは読めない（空の履歴になる）し、書けない。
	got, err := chatAs(t.Context(), c, "bob", first.result.GetSessionId())
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}
	if got.result.GetHistorySaved() {
		t.Error("他人のセッションに書き込めた")
	}
}

func TestChatRequiresCaller(t *testing.T) {
	t.Parallel()
	c := newAgentServer(t, localSessions(t))

	_, err := chatAs(t.Context(), c, "", "")
	if connect.CodeOf(err) != connect.CodeUnauthenticated {
		t.Errorf("code = %v, want Unauthenticated", connect.CodeOf(err))
	}
	if _, err := c.ListAgents(t.Context(), connect.NewRequest(&agentv1.ListAgentsRequest{})); err != nil {
		t.Errorf("一覧は利用者なしで呼べるはず: %v", err)
	}
}

func TestChatReportsHistoryNotSaved(t *testing.T) {
	t.Parallel()
	c := newAgentServer(t, failingAppend{localSessions(t)})

	got, err := chatAs(t.Context(), c, "alice", "s1")
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}
	if got.result.GetHistorySaved() {
		t.Error("保存に失敗したのに history_saved が true")
	}
	if got.result.GetAnswer() != "ok" {
		t.Errorf("保存の失敗で回答が失われた: %q", got.result.GetAnswer())
	}
}

func TestExecuteConfirmedToolCallCodes(t *testing.T) {
	t.Parallel()
	c := newAgentServer(t, localSessions(t))

	testCases := map[string]struct {
		toolCallID string
		want       connect.Code
	}{
		"ID が空なら InvalidArgument": {toolCallID: "", want: connect.CodeInvalidArgument},
		"承認待ちが無ければ NotFound":      {toolCallID: "req-1", want: connect.CodeNotFound},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			req := connect.NewRequest(&agentv1.ExecuteConfirmedToolCallRequest{ToolCallId: tc.toolCallID})
			req.Header().Set(libconnect.UserHeader, "alice")
			_, err := c.ExecuteConfirmedToolCall(t.Context(), req)
			if got := connect.CodeOf(err); got != tc.want {
				t.Errorf("code = %v, want %v", got, tc.want)
			}
		})
	}
}

func newBudgetServer(t *testing.T, limits libbudget.Limits, tokensPerCall int) agentv1connect.AgentServiceClient {
	t.Helper()
	var opts []usecase.Option
	if limits.Enabled() {
		opts = append(opts, usecase.WithBudget(libbudget.NewTracker(limits)))
	}
	return newAgentServerWith(t, stubAgent{tokensPerCall: tokensPerCall}, inmemory.NewSessions(), opts...)
}

func chat(t *testing.T, c agentv1connect.AgentServiceClient, sessionID string) error {
	t.Helper()
	_, err := chatAs(t.Context(), c, "alice", sessionID)
	return err
}

func TestChatWithoutBudget(t *testing.T) {
	c := newBudgetServer(t, libbudget.Limits{}, 1000)

	for i := range 3 {
		if err := chat(t, c, "s1"); err != nil {
			t.Fatalf("%d 回目で失敗した: %v", i+1, err)
		}
	}
}

func TestChatSessionBudgetExceeded(t *testing.T) {
	c := newBudgetServer(t, libbudget.Limits{SessionTokens: 100}, 150)

	if err := chat(t, c, "s1"); err != nil {
		t.Fatalf("1 回目は通るはずが失敗した: %v", err)
	}

	err := chat(t, c, "s1")
	if err == nil {
		t.Fatal("2 回目は拒否されるはずが通った")
	}
	if got := connect.CodeOf(err); got != connect.CodeResourceExhausted {
		t.Fatalf("コードが違う: got %v, want %v", got, connect.CodeResourceExhausted)
	}
}

func TestChatSessionBudgetIsPerSession(t *testing.T) {
	c := newBudgetServer(t, libbudget.Limits{SessionTokens: 100}, 150)

	if err := chat(t, c, "s1"); err != nil {
		t.Fatalf("s1 の 1 回目が失敗した: %v", err)
	}
	if err := chat(t, c, "s1"); err == nil {
		t.Fatal("s1 の 2 回目は拒否されるはず")
	}
	if err := chat(t, c, "s2"); err != nil {
		t.Fatalf("別セッション s2 は通るはずが失敗した: %v", err)
	}
}

func TestChatTotalBudgetExceeded(t *testing.T) {
	c := newBudgetServer(t, libbudget.Limits{TotalTokens: 100}, 150)

	if err := chat(t, c, "s1"); err != nil {
		t.Fatalf("1 回目が失敗した: %v", err)
	}
	if err := chat(t, c, "s2"); err == nil {
		t.Fatal("別セッションでも全体上限で拒否されるはず")
	}
}
