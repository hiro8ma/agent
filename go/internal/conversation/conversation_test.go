package conversation_test

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"sync"
	"testing"

	"connectrpc.com/connect"

	conversationv1 "github.com/hiro8ma/agent/go/gen/conversation/v1"
	"github.com/hiro8ma/agent/go/gen/conversation/v1/conversationv1connect"
	"github.com/hiro8ma/agent/go/internal/agentcore"
	"github.com/hiro8ma/agent/go/internal/conversation/adapter"
	"github.com/hiro8ma/agent/go/internal/conversation/client"
	"github.com/hiro8ma/agent/go/internal/conversation/domain"
	"github.com/hiro8ma/agent/go/internal/conversation/repository"
	"github.com/hiro8ma/agent/go/internal/conversation/usecase"
	"github.com/hiro8ma/agent/go/internal/lib/identity"
	"github.com/hiro8ma/agent/go/internal/lib/libconnect"
	"github.com/hiro8ma/agent/go/internal/lib/liberrors"
)

type env struct {
	store *client.SessionStore
	raw   conversationv1connect.ConversationServiceClient
}

func newEnv(t *testing.T) env {
	t.Helper()
	repo, err := repository.NewSQLite("file:" + filepath.Join(t.TempDir(), "conversation.db"))
	if err != nil {
		t.Fatalf("NewSQLite() error = %v", err)
	}
	path, h := adapter.NewHandler(usecase.New(repo), libconnect.HeaderAuthenticator)
	mux := http.NewServeMux()
	mux.Handle(path, h)
	srv := httptest.NewTestServer(t, mux)
	return env{
		store: client.NewSessionStore(srv.Client(), srv.URL),
		raw: conversationv1connect.NewConversationServiceClient(srv.Client(), srv.URL,
			connect.WithInterceptors(libconnect.ForwardIdentity())),
	}
}

func as(t *testing.T, user string) context.Context {
	t.Helper()
	return identity.With(t.Context(), identity.UserID(user))
}

func turn(i int) []agentcore.Message {
	return []agentcore.Message{
		{Role: "user", Text: fmt.Sprintf("質問 %d", i)},
		{Role: "model", Text: fmt.Sprintf("回答 %d", i)},
	}
}

func codeOf(err error) liberrors.Code {
	return liberrors.Convert(err).Code
}

func TestSessionIsVisibleOnlyToItsOwner(t *testing.T) {
	t.Parallel()
	e := newEnv(t)

	id, err := e.store.Create(as(t, "alice"), "research")
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	if err := e.store.Append(as(t, "alice"), id, turn(1)...); err != nil {
		t.Fatalf("Append() error = %v", err)
	}

	got, err := e.store.Load(as(t, "alice"), id)
	if err != nil || len(got) != 2 {
		t.Fatalf("所有者の Load() = %v, %v, want 2 件", got, err)
	}

	// 他人からは、無いセッションと区別がつかない。
	got, err = e.store.Load(as(t, "bob"), id)
	if err != nil || len(got) != 0 {
		t.Errorf("他人の Load() = %v, %v, want 空", got, err)
	}
	if err := e.store.Append(as(t, "bob"), id, turn(2)...); codeOf(err) != liberrors.CodeNotFound {
		t.Errorf("他人の Append() code = %v, want NOT_FOUND", codeOf(err))
	}
	_, err = e.raw.SubmitFeedback(as(t, "bob"), connect.NewRequest(&conversationv1.SubmitFeedbackRequest{
		SessionId: id, MessageId: "any", Rating: conversationv1.Rating_RATING_BAD,
	}))
	if connect.CodeOf(err) != connect.CodeNotFound {
		t.Errorf("他人の SubmitFeedback() code = %v, want NotFound", connect.CodeOf(err))
	}
}

func TestAppendCreatesSessionOwnedByFirstWriter(t *testing.T) {
	t.Parallel()
	e := newEnv(t)

	if err := e.store.Append(as(t, "alice"), "client-chosen-id", turn(1)...); err != nil {
		t.Fatalf("Append() error = %v", err)
	}
	if err := e.store.Append(as(t, "bob"), "client-chosen-id", turn(2)...); codeOf(err) != liberrors.CodeNotFound {
		t.Errorf("後から来た利用者の Append() code = %v, want NOT_FOUND", codeOf(err))
	}
}

func TestListMessagesPagesWithoutSkippingOrDuplicating(t *testing.T) {
	t.Parallel()
	e := newEnv(t)
	ctx := as(t, "alice")

	id, err := e.store.Create(ctx, "research")
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	for i := range 60 {
		if err := e.store.Append(ctx, id, turn(i)...); err != nil {
			t.Fatalf("Append() error = %v", err)
		}
	}

	var texts []string
	token := ""
	var sizes []int
	for page := 0; ; page++ {
		res, err := e.raw.ListMessages(ctx, connect.NewRequest(&conversationv1.ListMessagesRequest{
			SessionId: id, PageSize: 50, PageToken: token,
		}))
		if err != nil {
			t.Fatalf("ListMessages() error = %v", err)
		}
		sizes = append(sizes, len(res.Msg.GetMessages()))
		for _, m := range res.Msg.GetMessages() {
			texts = append(texts, m.GetText())
		}
		// 読んでいる途中に追記されても、既に読んだ位置はずれない。
		if page == 0 {
			if err := e.store.Append(ctx, id, turn(60)...); err != nil {
				t.Fatalf("Append() error = %v", err)
			}
		}
		token = res.Msg.GetNextPageToken()
		if token == "" {
			break
		}
	}

	if fmt.Sprint(sizes) != "[50 50 22]" {
		t.Errorf("ページごとの件数 = %v, want [50 50 22]", sizes)
	}
	for i := range 61 {
		if texts[2*i] != fmt.Sprintf("質問 %d", i) || texts[2*i+1] != fmt.Sprintf("回答 %d", i) {
			t.Fatalf("%d 往復目が順番どおりでない: %q %q", i, texts[2*i], texts[2*i+1])
		}
	}
}

func TestConcurrentAppendsKeepEveryMessage(t *testing.T) {
	t.Parallel()
	e := newEnv(t)
	ctx := as(t, "alice")

	id, err := e.store.Create(ctx, "research")
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	const writers = 20
	var wg sync.WaitGroup
	errs := make(chan error, writers)
	for i := range writers {
		wg.Go(func() { errs <- e.store.Append(ctx, id, turn(i)...) })
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatalf("同時の Append() error = %v", err)
		}
	}
	got, err := e.store.Load(ctx, id)
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if len(got) != 2*writers {
		t.Fatalf("残ったメッセージ = %d 件, want %d", len(got), 2*writers)
	}
	// 1 往復の 2 件は、他の書き込みに割り込まれず隣り合う。
	for i := 0; i < len(got); i += 2 {
		if got[i].Role != "user" || got[i+1].Role != "model" {
			t.Fatalf("%d 件目から往復が崩れた: %s %s", i, got[i].Role, got[i+1].Role)
		}
	}
}

func TestSubmitFeedback(t *testing.T) {
	t.Parallel()
	e := newEnv(t)
	ctx := as(t, "alice")

	id, err := e.store.Create(ctx, "research")
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	res, err := e.raw.AppendTurn(ctx, connect.NewRequest(&conversationv1.AppendTurnRequest{
		SessionId: id, Messages: []*conversationv1.Message{
			{Role: conversationv1.Role_ROLE_USER, Text: "質問"},
			{Role: conversationv1.Role_ROLE_MODEL, Text: "回答"},
		},
	}))
	if err != nil {
		t.Fatalf("AppendTurn() error = %v", err)
	}
	userMsg, modelMsg := res.Msg.GetMessages()[0].GetId(), res.Msg.GetMessages()[1].GetId()

	testCases := map[string]struct {
		messageID string
		rating    conversationv1.Rating
		want      connect.Code
	}{
		"アシスタントの回答は評価できる":    {messageID: modelMsg, rating: conversationv1.Rating_RATING_GOOD},
		"評価は付け直せる":           {messageID: modelMsg, rating: conversationv1.Rating_RATING_BAD},
		"利用者の発話は評価できない":      {messageID: userMsg, rating: conversationv1.Rating_RATING_GOOD, want: connect.CodeInvalidArgument},
		"評価が未指定なら拒否":         {messageID: modelMsg, want: connect.CodeInvalidArgument},
		"別のセッションのメッセージは無い扱い": {messageID: "missing", rating: conversationv1.Rating_RATING_GOOD, want: connect.CodeNotFound},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			_, err := e.raw.SubmitFeedback(ctx, connect.NewRequest(&conversationv1.SubmitFeedbackRequest{
				SessionId: id, MessageId: tc.messageID, Rating: tc.rating,
			}))
			if tc.want == 0 {
				if err != nil {
					t.Errorf("SubmitFeedback() error = %v", err)
				}
				return
			}
			if connect.CodeOf(err) != tc.want {
				t.Errorf("SubmitFeedback() code = %v, want %v", connect.CodeOf(err), tc.want)
			}
		})
	}
}

func TestRejectsBadRequests(t *testing.T) {
	t.Parallel()
	e := newEnv(t)
	id, err := e.store.Create(as(t, "alice"), "research")
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}

	testCases := map[string]struct {
		ctx  context.Context
		req  *conversationv1.ListMessagesRequest
		want connect.Code
	}{
		"利用者なしは拒否":           {ctx: t.Context(), req: &conversationv1.ListMessagesRequest{SessionId: id}, want: connect.CodeUnauthenticated},
		"壊れた page_token は拒否": {ctx: as(t, "alice"), req: &conversationv1.ListMessagesRequest{SessionId: id, PageToken: "zzz"}, want: connect.CodeInvalidArgument},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			_, err := e.raw.ListMessages(tc.ctx, connect.NewRequest(tc.req))
			if connect.CodeOf(err) != tc.want {
				t.Errorf("code = %v, want %v", connect.CodeOf(err), tc.want)
			}
		})
	}
}

func FuzzPageToken(f *testing.F) {
	for _, s := range []string{"", "djE6MA", "djE6LTE", "zzz", domain.EncodePageToken(42)} {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, token string) {
		seq, err := domain.DecodePageToken(token)
		if err != nil {
			return
		}
		if seq < 0 {
			t.Fatalf("負の位置 %d を受け付けた", seq)
		}
		if token != "" && domain.EncodePageToken(seq) != token {
			t.Fatalf("非正規の表現 %q を位置 %d として受け付けた", token, seq)
		}
	})
}
