package agentcore

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"time"

	"connectrpc.com/connect"
	"google.golang.org/protobuf/types/known/structpb"

	"github.com/hiro8ma/agent/go/gen/agent/v1"
	"github.com/hiro8ma/agent/go/gen/agent/v1/agentv1connect"
)

// Handler は Connect RPC（server streaming）で AgentService を公開する。
// 実装フレームワークには依存せず、Agent / SessionStore / ToolExecutor だけを見る。
type Handler struct {
	registry *Registry
	sessions SessionStore
	executor ToolExecutor
	logger   *slog.Logger
}

var _ agentv1connect.AgentServiceHandler = (*Handler)(nil)

func NewHandler(registry *Registry, sessions SessionStore, executor ToolExecutor, logger *slog.Logger) *Handler {
	return &Handler{registry: registry, sessions: sessions, executor: executor, logger: logger}
}

func (h *Handler) ListAgents(_ context.Context, _ *connect.Request[agentv1.ListAgentsRequest]) (*connect.Response[agentv1.ListAgentsResponse], error) {
	resp := &agentv1.ListAgentsResponse{}
	for _, info := range h.registry.List() {
		resp.Agents = append(resp.Agents, &agentv1.AgentInfo{Id: info.ID, Description: info.Description})
	}
	return connect.NewResponse(resp), nil
}

func (h *Handler) Ask(ctx context.Context, req *connect.Request[agentv1.AskRequest], stream *connect.ServerStream[agentv1.AskResponse]) error {
	msg := req.Msg
	if msg.GetAgentId() == "" || msg.GetSessionId() == "" || msg.GetMessage() == "" {
		return connect.NewError(connect.CodeInvalidArgument, errors.New("agentId, sessionId and message are required"))
	}
	a, ok := h.registry.Get(msg.GetAgentId())
	if !ok {
		return connect.NewError(connect.CodeNotFound, errors.New("unknown agent: "+msg.GetAgentId()))
	}

	history, err := h.sessions.Load(ctx, msg.GetSessionId())
	if err != nil {
		return connect.NewError(connect.CodeInternal, err)
	}

	start := time.Now()
	input := &AskInput{SessionID: msg.GetSessionId(), UserMessage: msg.GetMessage(), History: history}
	var final *AskOutput
	for chunk, out := range a.Ask(ctx, input) {
		if out != nil {
			final = out
			break
		}
		if err := stream.Send(&agentv1.AskResponse{AnswerDelta: chunk.AnswerDelta}); err != nil {
			return err
		}
	}
	if final == nil {
		return connect.NewError(connect.CodeInternal, errors.New("stream ended without result"))
	}

	h.logger.Info("ask_completed",
		"agentId", msg.GetAgentId(),
		"sessionId", msg.GetSessionId(),
		"latencyMs", time.Since(start).Milliseconds(),
		"inputTokens", final.Usage.InputTokens,
		"outputTokens", final.Usage.OutputTokens,
		"toolCalls", len(final.ToolCalls),
		"pendingToolCalls", len(final.PendingToolCalls),
		"finishReason", final.FinishReason,
		"error", final.ErrorMessage,
	)

	if err := stream.Send(&agentv1.AskResponse{Result: toResult(final)}); err != nil {
		return err
	}

	if final.ErrorMessage == "" {
		err := h.sessions.Append(ctx, msg.GetSessionId(),
			Message{Role: "user", Text: msg.GetMessage()},
			Message{Role: "model", Text: final.Answer},
		)
		if err != nil {
			h.logger.Error("failed to append session", "sessionId", msg.GetSessionId(), "error", err)
		}
	}
	return nil
}

func (h *Handler) ExecuteConfirmedToolCall(ctx context.Context, req *connect.Request[agentv1.ExecuteConfirmedToolCallRequest]) (*connect.Response[agentv1.ExecuteConfirmedToolCallResponse], error) {
	id := req.Msg.GetToolCallId()
	if id == "" {
		return nil, connect.NewError(connect.CodeInvalidArgument, errors.New("toolCallId is required"))
	}
	result, err := h.executor.Execute(ctx, id)
	if errors.Is(err, ErrPendingNotFound) {
		return nil, connect.NewError(connect.CodeNotFound, err)
	}
	if err != nil {
		return nil, connect.NewError(connect.CodeInternal, err)
	}
	h.logger.Info("tool_call_executed", "toolCallId", id)
	return connect.NewResponse(&agentv1.ExecuteConfirmedToolCallResponse{Result: toStruct(result)}), nil
}

func toResult(out *AskOutput) *agentv1.AskResult {
	result := &agentv1.AskResult{
		SessionId:    out.SessionID,
		Answer:       out.Answer,
		FinishReason: out.FinishReason,
		ErrorMessage: out.ErrorMessage,
		Usage: &agentv1.TokenUsage{
			InputTokens:  int32(out.Usage.InputTokens),
			OutputTokens: int32(out.Usage.OutputTokens),
			TotalTokens:  int32(out.Usage.TotalTokens),
		},
	}
	for _, tc := range out.ToolCalls {
		result.ToolCalls = append(result.ToolCalls, &agentv1.ToolCall{Name: tc.Name, Input: toStruct(tc.Input)})
	}
	for _, p := range out.PendingToolCalls {
		result.PendingToolCalls = append(result.PendingToolCalls, &agentv1.PendingToolCall{
			ToolCallId: p.ID,
			Name:       p.Name,
			Input:      toStruct(p.Input),
		})
	}
	return result
}

// toStruct は任意の値を JSON 経由で Struct へ変換する。変換できない値は nil。
func toStruct(v any) *structpb.Struct {
	if v == nil {
		return nil
	}
	b, err := json.Marshal(v)
	if err != nil {
		return nil
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		return nil
	}
	s, err := structpb.NewStruct(m)
	if err != nil {
		return nil
	}
	return s
}
