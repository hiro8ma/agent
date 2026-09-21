package knowledge_test

import (
	"context"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"connectrpc.com/connect"

	knowledgev1 "github.com/hiro8ma/agent/go/gen/knowledge/v1"
	"github.com/hiro8ma/agent/go/gen/knowledge/v1/knowledgev1connect"
	"github.com/hiro8ma/agent/go/internal/agentcore"
	"github.com/hiro8ma/agent/go/internal/knowledge/adapter"
	"github.com/hiro8ma/agent/go/internal/knowledge/client"
	"github.com/hiro8ma/agent/go/internal/lib/identity"
	"github.com/hiro8ma/agent/go/internal/lib/libconnect"
	"github.com/hiro8ma/agent/go/internal/lib/liberrors"
)

// recordingSearcher は受け取った件数と利用者を記録する。
type recordingSearcher struct {
	mu     sync.Mutex
	limits []int
	users  []identity.UserID
	err    error
}

func (r *recordingSearcher) Search(ctx context.Context, query string, limit int) ([]agentcore.KnowledgeDoc, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	id, _ := identity.From(ctx)
	r.limits = append(r.limits, limit)
	r.users = append(r.users, id)
	if r.err != nil {
		return nil, r.err
	}
	return []agentcore.KnowledgeDoc{{Title: "返品規程", Content: query + " への回答"}}, nil
}

func newServer(t *testing.T, s agentcore.KnowledgeSearcher) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()
	mux.Handle(adapter.NewHandler(s, libconnect.HeaderAuthenticator))
	return httptest.NewTestServer(t, mux)
}

func alice(t *testing.T) context.Context {
	t.Helper()
	return identity.With(t.Context(), "alice")
}

func TestSearcherCallsServiceAsCaller(t *testing.T) {
	t.Parallel()
	rec := &recordingSearcher{}
	srv := newServer(t, rec)

	docs, err := client.NewSearcher(srv.Client(), srv.URL).Search(alice(t), "返品", 3)
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	if len(docs) != 1 || docs[0].Content != "返品 への回答" {
		t.Errorf("docs = %+v", docs)
	}
	if rec.users[0] != "alice" || rec.limits[0] != 3 {
		t.Errorf("検索器が受けた利用者 = %q 件数 = %d", rec.users[0], rec.limits[0])
	}
}

func TestSearchValidatesInput(t *testing.T) {
	t.Parallel()
	rec := &recordingSearcher{}
	srv := newServer(t, rec)
	raw := knowledgev1connect.NewKnowledgeServiceClient(srv.Client(), srv.URL,
		connect.WithInterceptors(libconnect.ForwardIdentity()))

	testCases := map[string]struct {
		query     string
		limit     int32
		wantCode  connect.Code
		wantLimit int
	}{
		"件数が 0 なら既定の 5 件": {query: "返品", limit: 0, wantLimit: 5},
		"上限の 20 件は通す":     {query: "返品", limit: 20, wantLimit: 20},
		"上限を超える件数は拒否":     {query: "返品", limit: 21, wantCode: connect.CodeInvalidArgument},
		"負の件数は拒否":         {query: "返品", limit: -1, wantCode: connect.CodeInvalidArgument},
		"空白だけのクエリは拒否":     {query: "  ", wantCode: connect.CodeInvalidArgument},
		"長すぎるクエリは拒否":      {query: strings.Repeat("あ", 501), wantCode: connect.CodeInvalidArgument},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			_, err := raw.Search(alice(t), connect.NewRequest(&knowledgev1.SearchRequest{Query: tc.query, Limit: tc.limit}))
			if tc.wantCode != 0 {
				if connect.CodeOf(err) != tc.wantCode {
					t.Errorf("code = %v, want %v", connect.CodeOf(err), tc.wantCode)
				}
				return
			}
			if err != nil {
				t.Fatalf("Search() error = %v", err)
			}
		})
	}

	// 検索器まで届いた件数は、通ったケースの分だけ。
	rec.mu.Lock()
	defer rec.mu.Unlock()
	for _, l := range rec.limits {
		if l != 5 && l != 20 {
			t.Errorf("検証を通らないはずの件数 %d が検索器に届いた", l)
		}
	}
}

func TestSearchRequiresCaller(t *testing.T) {
	t.Parallel()
	srv := newServer(t, &recordingSearcher{})

	_, err := client.NewSearcher(srv.Client(), srv.URL).Search(t.Context(), "返品", 3)
	if got := liberrors.Convert(err).Code; got != liberrors.CodeUnauthenticated {
		t.Errorf("code = %v, want UNAUTHENTICATED", got)
	}
}

func TestSearchHidesBackendErrors(t *testing.T) {
	t.Parallel()
	srv := newServer(t, &recordingSearcher{err: errors.New("dial tcp 10.0.0.5:5432: password authentication failed")})
	raw := knowledgev1connect.NewKnowledgeServiceClient(srv.Client(), srv.URL,
		connect.WithInterceptors(libconnect.ForwardIdentity()))

	_, err := raw.Search(alice(t), connect.NewRequest(&knowledgev1.SearchRequest{Query: "返品"}))
	if connect.CodeOf(err) != connect.CodeInternal {
		t.Fatalf("code = %v, want Internal", connect.CodeOf(err))
	}
	if strings.Contains(err.Error(), "10.0.0.5") || strings.Contains(err.Error(), "password") {
		t.Errorf("検索器の内部の文言が外に出た: %v", err)
	}
}

func TestSearcherCapsLimit(t *testing.T) {
	t.Parallel()
	rec := &recordingSearcher{}
	srv := newServer(t, rec)

	if _, err := client.NewSearcher(srv.Client(), srv.URL).Search(alice(t), "返品", 100); err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	if rec.limits[0] != 20 {
		t.Errorf("検索器が受けた件数 = %d, want 20", rec.limits[0])
	}
}

func TestSearcherReportsUnavailableWhenServiceIsDown(t *testing.T) {
	t.Parallel()
	// NewTestServer の仮想ネットワークは Close 後の接続を拒否せずに待ち続けるので、閉じたループバックのポートを使う。
	l, err := (&net.ListenConfig{}).Listen(t.Context(), "tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("Listen() error = %v", err)
	}
	url := "http://" + l.Addr().String()
	if err := l.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}

	ctx, cancel := context.WithTimeout(alice(t), 5*time.Second)
	defer cancel()
	_, err = client.NewSearcher(http.DefaultClient, url).Search(ctx, "返品", 3)
	if got := liberrors.Convert(err).Code; got != liberrors.CodeUnavailable {
		t.Errorf("code = %v, want UNAVAILABLE (err %v)", got, err)
	}
}
