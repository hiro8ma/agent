package libconnect_test

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"connectrpc.com/connect"

	agentv1 "github.com/hiro8ma/agent/go/gen/agent/v1"
	"github.com/hiro8ma/agent/go/gen/agent/v1/agentv1connect"
	"github.com/hiro8ma/agent/go/internal/lib/identity"
	"github.com/hiro8ma/agent/go/internal/lib/libconnect"
	"github.com/hiro8ma/agent/go/internal/lib/liberrors"
)

// whoami は受け取った利用者を応答に写す。
type whoami struct {
	agentv1connect.UnimplementedAgentServiceHandler
}

func (whoami) ListAgents(ctx context.Context, _ *connect.Request[agentv1.ListAgentsRequest]) (*connect.Response[agentv1.ListAgentsResponse], error) {
	id, _ := identity.From(ctx)
	return connect.NewResponse(&agentv1.ListAgentsResponse{Agents: []*agentv1.AgentInfo{{Id: string(id)}}}), nil
}

func (whoami) Ask(ctx context.Context, _ *connect.Request[agentv1.AskRequest], stream *connect.ServerStream[agentv1.AskResponse]) error {
	id, err := identity.From(ctx)
	if err != nil {
		return libconnect.Error(err)
	}
	return stream.Send(&agentv1.AskResponse{AnswerDelta: string(id)})
}

func newServer(t *testing.T) *httptest.Server {
	t.Helper()
	path, h := agentv1connect.NewAgentServiceHandler(whoami{}, connect.WithInterceptors(
		libconnect.ServerIdentity(libconnect.HeaderAuthenticator, agentv1connect.AgentServiceListAgentsProcedure),
	))
	mux := http.NewServeMux()
	mux.Handle(path, h)
	return httptest.NewTestServer(t, mux)
}

func ask(ctx context.Context, c agentv1connect.AgentServiceClient, header string) (string, error) {
	req := connect.NewRequest(&agentv1.AskRequest{})
	if header != "" {
		req.Header().Set(libconnect.UserHeader, header)
	}
	stream, err := c.Ask(ctx, req)
	if err != nil {
		return "", err
	}
	defer stream.Close()
	var got string
	for stream.Receive() {
		got += stream.Msg().GetAnswerDelta()
	}
	return got, stream.Err()
}

func TestServerIdentity(t *testing.T) {
	t.Parallel()

	srv := newServer(t)
	client := agentv1connect.NewAgentServiceClient(srv.Client(), srv.URL)

	testCases := map[string]struct {
		header   string
		want     string
		wantCode connect.Code
	}{
		"ヘッダの利用者が context に入る": {header: "alice", want: "alice"},
		"ヘッダが無ければ止める":          {header: "", wantCode: connect.CodeUnauthenticated},
		"空白を含む利用者 ID は止める":     {header: "alice bob", wantCode: connect.CodeUnauthenticated},
		"長すぎる利用者 ID は止める":      {header: strings.Repeat("a", 129), wantCode: connect.CodeUnauthenticated},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			got, err := ask(t.Context(), client, tc.header)
			if tc.wantCode != 0 {
				if connect.CodeOf(err) != tc.wantCode {
					t.Fatalf("code = %v, want %v (err %v)", connect.CodeOf(err), tc.wantCode, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("Ask() error = %v", err)
			}
			if got != tc.want {
				t.Errorf("利用者 = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestServerIdentityLetsPublicProcedureThrough(t *testing.T) {
	t.Parallel()

	srv := newServer(t)
	client := agentv1connect.NewAgentServiceClient(srv.Client(), srv.URL)

	res, err := client.ListAgents(t.Context(), connect.NewRequest(&agentv1.ListAgentsRequest{}))
	if err != nil {
		t.Fatalf("ListAgents() error = %v", err)
	}
	if got := res.Msg.GetAgents()[0].GetId(); got != "" {
		t.Errorf("利用者なしの呼び出しに利用者 %q が入った", got)
	}
}

func TestForwardIdentityCarriesCallerToDownstream(t *testing.T) {
	t.Parallel()

	srv := newServer(t)
	client := agentv1connect.NewAgentServiceClient(srv.Client(), srv.URL,
		connect.WithInterceptors(libconnect.ForwardIdentity()))

	testCases := map[string]struct {
		ctx      context.Context
		want     string
		wantCode connect.Code
	}{
		"呼び出し元の利用者のまま下流を呼ぶ": {ctx: identity.With(t.Context(), "alice"), want: "alice"},
		"利用者が無ければ下流で止まる":    {ctx: t.Context(), wantCode: connect.CodeUnauthenticated},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			got, err := ask(tc.ctx, client, "")
			if tc.wantCode != 0 {
				if connect.CodeOf(err) != tc.wantCode {
					t.Fatalf("code = %v, want %v", connect.CodeOf(err), tc.wantCode)
				}
				return
			}
			if err != nil {
				t.Fatalf("Ask() error = %v", err)
			}
			if got != tc.want {
				t.Errorf("下流が受け取った利用者 = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestErrorHidesInternalMessages(t *testing.T) {
	t.Parallel()

	testCases := map[string]struct {
		err      error
		wantCode connect.Code
		wantMsg  string
	}{
		"Code 付きは対応する Code にする":   {err: liberrors.Newf(liberrors.CodeNotFound, "セッションが無い"), wantCode: connect.CodeNotFound, wantMsg: "セッションが無い"},
		"所有者違いは PermissionDenied": {err: liberrors.Newf(liberrors.CodePermissionDeny, "他人のセッション"), wantCode: connect.CodePermissionDenied, wantMsg: "他人のセッション"},
		"利用者なしは Unauthenticated":  {err: identity.ErrUnauthenticated, wantCode: connect.CodeUnauthenticated},
		"Code の無いエラーは文言を出さない":     {err: errors.New("dial tcp 10.0.0.1: refused"), wantCode: connect.CodeInternal, wantMsg: "内部エラー"},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			ce, ok := errors.AsType[*connect.Error](libconnect.Error(tc.err))
			if !ok {
				t.Fatalf("connect.Error ではない")
			}
			if ce.Code() != tc.wantCode {
				t.Errorf("code = %v, want %v", ce.Code(), tc.wantCode)
			}
			if tc.wantMsg != "" && ce.Message() != tc.wantMsg {
				t.Errorf("message = %q, want %q", ce.Message(), tc.wantMsg)
			}
		})
	}
}

func FuzzParseUserID(f *testing.F) {
	for _, s := range []string{"alice", "a.b@c-d_e", "", "a\nb", "ユーザー"} {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, s string) {
		id, err := identity.ParseUserID(s)
		if err != nil {
			return
		}
		// 通った値はヘッダに載せても壊れない。
		for _, r := range string(id) {
			if r < 0x21 || r > 0x7e {
				t.Fatalf("表示できない文字 %q を通した", r)
			}
		}
	})
}
