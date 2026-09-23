package action_test

import (
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"connectrpc.com/connect"

	actionv1 "github.com/hiro8ma/agent/go/gen/action/v1"
	"github.com/hiro8ma/agent/go/gen/action/v1/actionv1connect"
	agentv1 "github.com/hiro8ma/agent/go/gen/agent/v1"
	"github.com/hiro8ma/agent/go/gen/agent/v1/agentv1connect"
	"github.com/hiro8ma/agent/go/internal/action"
	"github.com/hiro8ma/agent/go/internal/action/adapter"
	"github.com/hiro8ma/agent/go/internal/action/client"
	"github.com/hiro8ma/agent/go/internal/agentcore"
	"github.com/hiro8ma/agent/go/internal/agentcore/backend"
	"github.com/hiro8ma/agent/go/internal/approval"
	"github.com/hiro8ma/agent/go/internal/lib/identity"
	"github.com/hiro8ma/agent/go/internal/lib/libconnect"
)

type env struct {
	actions actionv1connect.ActionServiceClient // 承認者などが直接呼ぶ口
	agents  agentv1connect.AgentServiceClient   // 依頼者がエージェント経由で実行する口
	gate    *client.Remote                      // エージェントのツールが使う窓口
	orders  *backend.InMemoryOrders
}

// newEnv は ActionService と AgentService を別々のサーバとして立てる。
func newEnv(t *testing.T) env {
	t.Helper()
	svc := approval.NewService(action.DefaultRules(), []string{"manager", "manager2"}, time.Hour)
	actionMux := http.NewServeMux()
	actionMux.Handle(adapter.NewHandler(svc, libconnect.HeaderAuthenticator))
	actionSrv := httptest.NewTestServer(t, actionMux)
	gate := client.NewRemote(actionSrv.Client(), actionSrv.URL)

	orders := backend.NewInMemoryOrders()
	h := agentcore.NewHandler(agentcore.NewRegistry(), nil, action.Executor{Gate: gate, Orders: orders}, slog.New(slog.DiscardHandler))
	agentMux := http.NewServeMux()
	agentMux.Handle(agentcore.NewConnectHandler(h, libconnect.HeaderAuthenticator))
	agentSrv := httptest.NewTestServer(t, agentMux)

	return env{
		actions: actionv1connect.NewActionServiceClient(actionSrv.Client(), actionSrv.URL),
		agents:  agentv1connect.NewAgentServiceClient(agentSrv.Client(), agentSrv.URL),
		gate:    gate,
		orders:  orders,
	}
}

func as[T any](user string, msg *T) *connect.Request[T] {
	req := connect.NewRequest(msg)
	req.Header().Set(libconnect.UserHeader, user)
	return req
}

func (e env) execute(t *testing.T, user, id string) (*connect.Response[agentv1.ExecuteConfirmedToolCallResponse], error) {
	t.Helper()
	return e.agents.ExecuteConfirmedToolCall(t.Context(), as(user, &agentv1.ExecuteConfirmedToolCallRequest{ToolCallId: id}))
}

// requestChange はエージェントのツールと同じ経路で、alice の依頼として支払い方法の変更を頼む。
func (e env) requestChange(t *testing.T, method string) string {
	t.Helper()
	ctx := identity.With(t.Context(), "alice")
	out, err := action.RequestPaymentChange(ctx, e.gate, e.orders, "ord-001", method)
	if err != nil {
		t.Fatalf("RequestPaymentChange() error = %v", err)
	}
	p, ok := action.PendingFromResult(action.ToolUpdatePaymentMethod, out)
	if !ok {
		t.Fatalf("承認待ちにならない: %v", out)
	}
	return p.ID
}

func TestApprovalFlowAcrossServices(t *testing.T) {
	t.Parallel()
	e := newEnv(t)
	id := e.requestChange(t, "クレジットカード")

	// 承認前は実行できない。
	if _, err := e.execute(t, "alice", id); connect.CodeOf(err) != connect.CodeFailedPrecondition {
		t.Fatalf("承認前の実行 code = %v", connect.CodeOf(err))
	}

	// 依頼者本人と、承認者でない利用者は承認できない。
	for _, who := range []string{"alice", "mallory"} {
		_, err := e.actions.Decide(t.Context(), as(who, &actionv1.DecideRequest{RequestId: id, Approve: true}))
		if connect.CodeOf(err) != connect.CodePermissionDenied {
			t.Errorf("%s の承認 code = %v", who, connect.CodeOf(err))
		}
	}

	// 承認者には承認待ちとして見え、依頼者には見えない。
	list := func(user string) int {
		res, err := e.actions.ListRequests(t.Context(), as(user, &actionv1.ListRequestsRequest{View: actionv1.View_VIEW_TO_APPROVE}))
		if err != nil {
			t.Fatal(err)
		}
		return len(res.Msg.GetRequests())
	}
	if list("manager") != 1 || list("alice") != 0 {
		t.Errorf("承認待ちの見え方 manager=%d alice=%d", list("manager"), list("alice"))
	}

	if _, err := e.actions.Decide(t.Context(), as("manager", &actionv1.DecideRequest{RequestId: id, Approve: true})); err != nil {
		t.Fatalf("Decide() error = %v", err)
	}

	// 他人は、承認済みの依頼を実行できない。
	if _, err := e.execute(t, "mallory", id); connect.CodeOf(err) != connect.CodePermissionDenied {
		t.Errorf("他人の実行 code = %v", connect.CodeOf(err))
	}
	// 依頼者は、承認された引数で 1 回だけ実行できる。
	res, err := e.execute(t, "alice", id)
	if err != nil {
		t.Fatalf("実行 error = %v", err)
	}
	if got := res.Msg.GetResult().AsMap()["order"].(map[string]any)["paymentMethod"]; got != "クレジットカード" {
		t.Errorf("変更後の支払い方法 = %v", got)
	}
	if _, err := e.execute(t, "alice", id); connect.CodeOf(err) != connect.CodeFailedPrecondition {
		t.Errorf("2 回目の実行 code = %v", connect.CodeOf(err))
	}

	// 監査ログに、依頼、承認、実行、他人の実行の試みが残る。
	audit, err := e.actions.ListAudit(t.Context(), as("manager", &actionv1.ListAuditRequest{RequestId: id}))
	if err != nil {
		t.Fatal(err)
	}
	var got []string
	for _, a := range audit.Msg.GetEntries() {
		got = append(got, a.GetAction()+":"+a.GetActor())
	}
	want := []string{"open:alice", "approved:manager", "take_denied:mallory", "take:alice"}
	if len(got) != len(want) {
		t.Fatalf("監査ログ = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("監査ログ[%d] = %s, want %s", i, got[i], want[i])
		}
	}
}

func TestRejectedRequestCannotBeExecuted(t *testing.T) {
	t.Parallel()
	e := newEnv(t)
	id := e.requestChange(t, "代引き")
	if _, err := e.actions.Decide(t.Context(), as("manager2", &actionv1.DecideRequest{RequestId: id, Approve: false})); err != nil {
		t.Fatal(err)
	}
	if _, err := e.execute(t, "alice", id); connect.CodeOf(err) != connect.CodeFailedPrecondition {
		t.Errorf("却下された依頼の実行 code = %v", connect.CodeOf(err))
	}
	o, _ := e.orders.GetOrder(t.Context(), "ord-001")
	if o.PaymentMethod != "銀行振込" {
		t.Errorf("却下したのに支払い方法が変わった: %s", o.PaymentMethod)
	}
}

func TestUnknownRequestIsNotFound(t *testing.T) {
	t.Parallel()
	e := newEnv(t)
	if _, err := e.execute(t, "alice", "missing"); connect.CodeOf(err) != connect.CodeNotFound {
		t.Errorf("code = %v", connect.CodeOf(err))
	}
}

func TestAuditIsNotReadableByOthers(t *testing.T) {
	t.Parallel()
	e := newEnv(t)
	id := e.requestChange(t, "クレジットカード")
	_, err := e.actions.ListAudit(t.Context(), as("mallory", &actionv1.ListAuditRequest{RequestId: id}))
	if connect.CodeOf(err) != connect.CodePermissionDenied {
		t.Errorf("code = %v", connect.CodeOf(err))
	}
}
