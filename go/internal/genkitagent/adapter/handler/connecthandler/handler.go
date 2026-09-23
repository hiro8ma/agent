// Package connecthandler は AgentService を ConnectRPC（server streaming）で公開する。
package connecthandler

import (
	"context"
	"errors"
	"net/http"

	"connectrpc.com/connect"

	agentv1 "github.com/hiro8ma/agent/go/gen/agent/v1"
	"github.com/hiro8ma/agent/go/gen/agent/v1/agentv1connect"
	"github.com/hiro8ma/agent/go/internal/genkitagent/usecase"
	"github.com/hiro8ma/agent/go/internal/lib/libbudget"
	"github.com/hiro8ma/agent/go/internal/lib/libconnect"
)

type Handler struct {
	agents *usecase.AgentService
}

var _ agentv1connect.AgentServiceHandler = (*Handler)(nil)

func New(agents *usecase.AgentService) *Handler {
	return &Handler{agents: agents}
}

// Route は利用者の特定まで含めて公開する。一覧取得だけは利用者なしで呼べる。
func Route(h *Handler, auth libconnect.Authenticator) (string, http.Handler) {
	return agentv1connect.NewAgentServiceHandler(h, connect.WithInterceptors(
		// span を認証の外側に置き、利用者を特定できずに拒否した呼び出しも記録する。
		libconnect.EdgeTelemetry(),
		libconnect.ServerIdentity(auth, agentv1connect.AgentServiceListAgentsProcedure),
	))
}

func (h *Handler) ListAgents(ctx context.Context, _ *connect.Request[agentv1.ListAgentsRequest]) (*connect.Response[agentv1.ListAgentsResponse], error) {
	res, err := h.agents.ListAgents(ctx, &usecase.ListAgentsRequest{})
	if err != nil {
		return nil, toConnectError(err)
	}
	return connect.NewResponse(toListAgentsResponse(res)), nil
}

func (h *Handler) Chat(ctx context.Context, req *connect.Request[agentv1.ChatRequest], stream *connect.ServerStream[agentv1.ChatResponse]) error {
	for res, err := range h.agents.Chat(ctx, fromChatRequest(req.Msg)) {
		if err != nil {
			return toConnectError(err)
		}
		if err := stream.Send(toChatResponse(res)); err != nil {
			return err
		}
	}
	return nil
}

func (h *Handler) ExecuteConfirmedToolCall(ctx context.Context, req *connect.Request[agentv1.ExecuteConfirmedToolCallRequest]) (*connect.Response[agentv1.ExecuteConfirmedToolCallResponse], error) {
	res, err := h.agents.ExecuteConfirmedToolCall(ctx, &usecase.ExecuteConfirmedToolCallRequest{ToolCallID: req.Msg.GetToolCallId()})
	if err != nil {
		return nil, toConnectError(err)
	}
	return connect.NewResponse(&agentv1.ExecuteConfirmedToolCallResponse{Result: toStruct(res.Result)}), nil
}

func toConnectError(err error) error {
	if e, ok := errors.AsType[*libbudget.ExceededError](err); ok {
		return connect.NewError(connect.CodeResourceExhausted, e)
	}
	return libconnect.Error(err)
}
