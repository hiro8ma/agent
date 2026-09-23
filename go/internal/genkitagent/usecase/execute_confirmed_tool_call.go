package usecase

import (
	"context"
	"errors"
	"fmt"

	"github.com/hiro8ma/agent/go/internal/genkitagent/domain/model"
	"github.com/hiro8ma/agent/go/internal/lib/liberrors"
)

// ExecuteConfirmedToolCall は承認済みの依頼を、承認された引数で実行する。モデルに呼び直させない。
func (s *AgentService) ExecuteConfirmedToolCall(ctx context.Context, req *ExecuteConfirmedToolCallRequest) (*ExecuteConfirmedToolCallResponse, error) {
	if req.ToolCallID == "" {
		return nil, liberrors.Newf(liberrors.CodeInvalidArgument, "toolCallId is required")
	}
	// 他人の依頼、未承認、期限切れは、承認の窓口が付けた Code のまま返す。
	action, err := s.gate.Take(ctx, req.ToolCallID)
	if errors.Is(err, model.ErrPendingNotFound) {
		return nil, liberrors.Wrap(liberrors.CodeNotFound, err, "承認待ちの依頼が無い")
	}
	if err != nil {
		return nil, err
	}
	result, err := s.execute(ctx, action)
	if err != nil {
		return nil, err
	}
	s.logger.InfoContext(ctx, "tool_call_executed", "toolCallId", req.ToolCallID)
	return &ExecuteConfirmedToolCallResponse{Result: result}, nil
}

func (s *AgentService) execute(ctx context.Context, action *model.ApprovedAction) (map[string]any, error) {
	switch action.Tool {
	case model.ToolUpdatePaymentMethod:
		orderID, _ := action.Args["orderId"].(string)
		method, _ := action.Args["paymentMethod"].(string)
		order, err := s.orders.UpdatePaymentMethod(ctx, orderID, method)
		if err != nil {
			return nil, err
		}
		return map[string]any{"order": order}, nil
	}
	return nil, fmt.Errorf("action: 実行の手順が無いツール %q", action.Tool)
}
