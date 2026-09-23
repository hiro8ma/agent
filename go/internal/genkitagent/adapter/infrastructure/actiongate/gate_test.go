package actiongate_test

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
	actionclient "github.com/hiro8ma/agent/go/internal/action/client"
	"github.com/hiro8ma/agent/go/internal/approval"
	"github.com/hiro8ma/agent/go/internal/genkitagent/adapter/handler/connecthandler"
	"github.com/hiro8ma/agent/go/internal/genkitagent/adapter/infrastructure/actiongate"
	"github.com/hiro8ma/agent/go/internal/genkitagent/adapter/infrastructure/inmemory"
	"github.com/hiro8ma/agent/go/internal/genkitagent/domain/model"
	"github.com/hiro8ma/agent/go/internal/genkitagent/domain/service"
	"github.com/hiro8ma/agent/go/internal/genkitagent/usecase"
	"github.com/hiro8ma/agent/go/internal/lib/identity"
	"github.com/hiro8ma/agent/go/internal/lib/libconnect"
)

type env struct {
	actions actionv1connect.ActionServiceClient
	agents  agentv1connect.AgentServiceClient
	gate    *actiongate.Gate
	orders  *inmemory.Orders
}

func newEnv(t *testing.T) env {
	t.Helper()
	svc := approval.NewService(action.DefaultRules(), []string{"manager"}, time.Hour)
	actionMux := http.NewServeMux()
	actionMux.Handle(adapter.NewHandler(svc, libconnect.HeaderAuthenticator))
	actionSrv := httptest.NewTestServer(t, actionMux)
	gate := actiongate.New(actionclient.NewRemote(actionSrv.Client(), actionSrv.URL))

	orders := inmemory.NewOrders()
	uc := usecase.NewAgentService(service.NewRegistry(), inmemory.NewSessions(), gate, orders, slog.New(slog.DiscardHandler))
	agentMux := http.NewServeMux()
	agentMux.Handle(connecthandler.Route(connecthandler.New(uc), libconnect.HeaderAuthenticator))
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

func TestApprovalFlowThroughGenkitAgentService(t *testing.T) {
	t.Parallel()
	e := newEnv(t)

	d, err := e.gate.Authorize(identity.With(t.Context(), "alice"), model.ToolUpdatePaymentMethod,
		map[string]any{"orderId": "ord-001", "paymentMethod": "クレジットカード"})
	if err != nil {
		t.Fatalf("Authorize() error = %v", err)
	}
	if d.Outcome != model.ActionNeedsApproval || d.RequestID == "" || d.Risk != approval.High.String() {
		t.Fatalf("Authorize() = %+v, want 承認待ち", d)
	}

	if _, err := e.execute(t, "alice", d.RequestID); connect.CodeOf(err) != connect.CodeFailedPrecondition {
		t.Fatalf("承認前の実行 code = %v", connect.CodeOf(err))
	}
	if _, err := e.actions.Decide(t.Context(), as("manager", &actionv1.DecideRequest{RequestId: d.RequestID, Approve: true})); err != nil {
		t.Fatalf("Decide() error = %v", err)
	}
	res, err := e.execute(t, "alice", d.RequestID)
	if err != nil {
		t.Fatalf("実行 error = %v", err)
	}
	if got := res.Msg.GetResult().AsMap()["order"].(map[string]any)["paymentMethod"]; got != "クレジットカード" {
		t.Errorf("変更後の支払い方法 = %v", got)
	}
}

func TestUnknownRequestIsNotFound(t *testing.T) {
	t.Parallel()
	e := newEnv(t)
	if _, err := e.execute(t, "alice", "missing"); connect.CodeOf(err) != connect.CodeNotFound {
		t.Errorf("code = %v, want NotFound", connect.CodeOf(err))
	}
}
