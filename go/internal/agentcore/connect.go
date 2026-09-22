package agentcore

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"time"

	"connectrpc.com/connect"
	"google.golang.org/protobuf/types/known/structpb"

	agentv1 "github.com/hiro8ma/agent/go/gen/agent/v1"
	"github.com/hiro8ma/agent/go/gen/agent/v1/agentv1connect"
	"github.com/hiro8ma/agent/go/internal/lib/libconnect"
)

// Handler は Connect RPC（server streaming）で AgentService を公開する。
// 実装フレームワークには依存せず、Agent / SessionStore / ToolExecutor だけを見る。
type Handler struct {
	registry *Registry
	sessions SessionStore
	executor ToolExecutor
	logger   *slog.Logger
	budget   *BudgetTracker // nil ならトークン予算のチェックなし
}

var _ agentv1connect.AgentServiceHandler = (*Handler)(nil)

// NewConnectHandler は利用者の特定まで含めて公開する。一覧取得だけは利用者なしで呼べる。
func NewConnectHandler(h *Handler, auth libconnect.Authenticator) (string, http.Handler) {
	return agentv1connect.NewAgentServiceHandler(h, connect.WithInterceptors(
		// span を認証の外側に置き、利用者を特定できずに拒否した呼び出しも記録する。
		libconnect.EdgeTelemetry(),
		libconnect.ServerIdentity(auth, agentv1connect.AgentServiceListAgentsProcedure),
	))
}

func NewHandler(registry *Registry, sessions SessionStore, executor ToolExecutor, logger *slog.Logger) *Handler {
	return &Handler{registry: registry, sessions: sessions, executor: executor, logger: logger}
}

// WithBudget はトークン予算を有効にする。
func (h *Handler) WithBudget(b *BudgetTracker) *Handler {
	h.budget = b
	return h
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
	if msg.GetAgentId() == "" || msg.GetMessage() == "" {
		return connect.NewError(connect.CodeInvalidArgument, errors.New("agentId and message are required"))
	}
	a, ok := h.registry.Get(msg.GetAgentId())
	if !ok {
		return connect.NewError(connect.CodeNotFound, errors.New("unknown agent: "+msg.GetAgentId()))
	}

	sessionID, err := h.resolveSession(ctx, msg)
	if err != nil {
		return err
	}

	if err := h.budget.Check(sessionID); err != nil {
		return connect.NewError(connect.CodeResourceExhausted, err)
	}

	// 他人のセッションは保管先が拒否する。拒否の理由はそのまま Code に写す。
	history, err := h.sessions.Load(ctx, sessionID)
	if err != nil {
		return libconnect.Error(err)
	}

	start := time.Now()
	input := &AskInput{SessionID: sessionID, UserMessage: msg.GetMessage(), History: history}
	var final *AskOutput
	for chunk, out := range a.Ask(ctx, input) {
		if out != nil {
			final = out
			break
		}
		if err := stream.Send(&agentv1.AskResponse{Event: &agentv1.AskResponse_AnswerDelta{AnswerDelta: chunk.AnswerDelta}}); err != nil {
			return err
		}
	}
	if final == nil {
		return connect.NewError(connect.CodeInternal, errors.New("stream ended without result"))
	}
	final.SessionID = sessionID

	h.budget.Add(sessionID, final.Usage)

	// 最終応答を送る前に保存し、保存できたかを応答に載せる。送った後に保存すると、失敗を呼び出し元に伝えられない。
	historySaved := false
	if final.ErrorMessage == "" {
		err := h.sessions.Append(ctx, sessionID,
			Message{Role: "user", Text: msg.GetMessage()},
			Message{Role: "model", Text: final.Answer},
		)
		historySaved = err == nil
		if err != nil {
			h.logger.Error("failed to append session", "sessionId", sessionID, "error", err)
		}
	}

	attrs := []any{
		"agentId", msg.GetAgentId(),
		"sessionId", sessionID,
		"latencyMs", time.Since(start).Milliseconds(),
		"inputTokens", final.Usage.InputTokens,
		"outputTokens", final.Usage.OutputTokens,
		"toolCalls", len(final.ToolCalls),
		"pendingToolCalls", len(final.PendingToolCalls),
		"finishReason", final.FinishReason,
		"historySaved", historySaved,
		"error", final.ErrorMessage,
	}
	if h.budget != nil {
		sessionUsed, totalUsed := h.budget.Used(sessionID)
		attrs = append(attrs, "budget_session_used", sessionUsed, "budget_total_used", totalUsed)
	}
	h.logger.Info("ask_completed", attrs...)

	result := toResult(final)
	result.HistorySaved = historySaved
	return stream.Send(&agentv1.AskResponse{Event: &agentv1.AskResponse_Result{Result: result}})
}

// resolveSession は session_id が空なら、保管先に新しいセッションを作らせる。
func (h *Handler) resolveSession(ctx context.Context, msg *agentv1.AskRequest) (string, error) {
	if id := msg.GetSessionId(); id != "" {
		return id, nil
	}
	creator, ok := h.sessions.(SessionCreator)
	if !ok {
		return "", connect.NewError(connect.CodeInvalidArgument, errors.New("sessionId is required"))
	}
	id, err := creator.Create(ctx, msg.GetAgentId())
	if err != nil {
		return "", libconnect.Error(err)
	}
	return id, nil
}

func (h *Handler) ExecuteConfirmedToolCall(ctx context.Context, req *connect.Request[agentv1.ExecuteConfirmedToolCallRequest]) (*connect.Response[agentv1.ExecuteConfirmedToolCallResponse], error) {
	id := req.Msg.GetToolCallId()
	if id == "" {
		return nil, connect.NewError(connect.CodeInvalidArgument, errors.New("toolCallId is required"))
	}
	// 他人の依頼、未承認、期限切れは、承認の窓口が付けた Code のまま返す。
	result, err := h.executor.Execute(ctx, id)
	if errors.Is(err, ErrPendingNotFound) {
		return nil, connect.NewError(connect.CodeNotFound, errors.New("承認待ちの依頼が無い"))
	}
	if err != nil {
		return nil, libconnect.Error(err)
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
