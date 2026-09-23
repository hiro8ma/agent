package usecase

import (
	"context"
	"iter"
	"log/slog"
	"time"

	"github.com/hiro8ma/agent/go/internal/genkitagent/domain/model"
	"github.com/hiro8ma/agent/go/internal/genkitagent/domain/repository"
	"github.com/hiro8ma/agent/go/internal/lib/liberrors"
)

func (s *AgentService) Chat(ctx context.Context, req *ChatRequest) iter.Seq2[*ChatResponse, error] {
	return func(yield func(*ChatResponse, error) bool) {
		if req.AgentID == "" || req.Message == "" {
			yield(nil, liberrors.Newf(liberrors.CodeInvalidArgument, "agentId and message are required"))
			return
		}
		a, ok := s.agents.Get(req.AgentID)
		if !ok {
			yield(nil, liberrors.Newf(liberrors.CodeNotFound, "unknown agent: %s", req.AgentID))
			return
		}
		sessionID, err := s.resolveSession(ctx, req)
		if err != nil {
			yield(nil, err)
			return
		}
		if err := s.budget.Check(sessionID); err != nil {
			yield(nil, err)
			return
		}
		// 他人のセッションは保存先が拒否する。
		history, err := s.sessions.ListMessages(ctx, sessionID)
		if err != nil {
			yield(nil, err)
			return
		}

		start := time.Now()
		input := &model.ChatInput{SessionID: sessionID, UserMessage: req.Message, History: history}
		var final *model.ChatOutput
		for chunk, out := range a.Chat(ctx, input) {
			if out != nil {
				final = out
				break
			}
			if !yield(&ChatResponse{Delta: chunk}, nil) {
				return
			}
		}
		if final == nil {
			yield(nil, liberrors.Newf(liberrors.CodeInternal, "stream ended without result"))
			return
		}
		final.SessionID = sessionID
		s.budget.Add(sessionID, final.Usage.Total())

		// 最終応答を返す前に保存し、保存できたかを応答に載せる。返した後に保存すると、失敗を呼び出し元に伝えられない。
		historySaved := false
		if !final.Failed() {
			err := s.sessions.CreateMessages(ctx, sessionID, []model.Message{
				{Role: model.RoleUser, Text: req.Message},
				{Role: model.RoleModel, Text: final.Answer},
			})
			historySaved = err == nil
			if err != nil {
				s.logger.ErrorContext(ctx, "failed to append session", "sessionId", sessionID, "error", err)
			}
		}
		s.logCompleted(ctx, req.AgentID, sessionID, time.Since(start), final, historySaved)

		yield(&ChatResponse{Output: final, HistorySaved: historySaved}, nil)
	}
}

func (s *AgentService) resolveSession(ctx context.Context, req *ChatRequest) (string, error) {
	if req.SessionID != "" {
		return req.SessionID, nil
	}
	creator, ok := s.sessions.(repository.SessionCreator)
	if !ok {
		return "", liberrors.Newf(liberrors.CodeInvalidArgument, "sessionId is required")
	}
	return creator.CreateSession(ctx, req.AgentID)
}

func (s *AgentService) logCompleted(ctx context.Context, agentID, sessionID string, latency time.Duration, final *model.ChatOutput, historySaved bool) {
	attrs := []any{
		"agentId", agentID,
		"sessionId", sessionID,
		"latencyMs", latency.Milliseconds(),
		"inputTokens", final.Usage.InputTokens,
		"outputTokens", final.Usage.OutputTokens,
		"toolCalls", len(final.ToolCalls),
		"pendingToolCalls", len(final.PendingToolCalls),
		"finishReason", final.FinishReason,
		"historySaved", historySaved,
		"error", final.ErrorMessage,
	}
	if s.budget != nil {
		sessionUsed, totalUsed := s.budget.Used(sessionID)
		attrs = append(attrs, "budget_session_used", sessionUsed, "budget_total_used", totalUsed)
	}
	level := slog.LevelInfo
	if final.Failed() {
		level = slog.LevelWarn
	}
	s.logger.Log(ctx, level, "chat_completed", attrs...)
}
