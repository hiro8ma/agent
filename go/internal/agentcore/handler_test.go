package agentcore_test

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"

	"connectrpc.com/connect"

	agentv1 "github.com/hiro8ma/agent/go/gen/agent/v1"
	"github.com/hiro8ma/agent/go/gen/agent/v1/agentv1connect"
	"github.com/hiro8ma/agent/go/internal/agentcore"
	"github.com/hiro8ma/agent/go/internal/conversation/client"
	"github.com/hiro8ma/agent/go/internal/conversation/repository"
	"github.com/hiro8ma/agent/go/internal/conversation/usecase"
	"github.com/hiro8ma/agent/go/internal/lib/libconnect"
)

// failingAppend は履歴の読み出しはできるが、追記に失敗する保管先。
type failingAppend struct{ agentcore.SessionStore }

func (failingAppend) Append(context.Context, string, ...agentcore.Message) error {
	return errors.New("disk full")
}

func newAgentServer(t *testing.T, sessions agentcore.SessionStore) agentv1connect.AgentServiceClient {
	t.Helper()
	h := agentcore.NewHandler(agentcore.NewRegistry(stubAgent{tokensPerCall: 1}), sessions, stubExecutor{}, slog.New(slog.DiscardHandler))
	mux := http.NewServeMux()
	mux.Handle(agentcore.NewConnectHandler(h, libconnect.HeaderAuthenticator))
	srv := httptest.NewTestServer(t, mux)
	return agentv1connect.NewAgentServiceClient(srv.Client(), srv.URL)
}

func localSessions(t *testing.T) *client.Local {
	t.Helper()
	repo, err := repository.NewSQLite("file:" + filepath.Join(t.TempDir(), "conversation.db"))
	if err != nil {
		t.Fatalf("NewSQLite() error = %v", err)
	}
	return client.NewLocal(usecase.New(repo))
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

	// 返った ID で続けられる。
	if _, err := chatAs(t.Context(), c, "alice", got.result.GetSessionId()); err != nil {
		t.Fatalf("2 回目の Chat() error = %v", err)
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
