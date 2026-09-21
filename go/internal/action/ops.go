package action

import (
	"context"
	"errors"
	"fmt"

	"github.com/hiro8ma/agent/go/internal/agentcore"
	"github.com/hiro8ma/agent/go/internal/approval"
	"github.com/hiro8ma/agent/go/internal/lib/liberrors"
)

// RequestPaymentChange は支払い方法の変更を依頼する。ADK と Genkit のツールはこれを呼ぶ。
// 結果はモデルが読むので、承認待ちなら依頼の ID と、何を待っているかを入れる。
func RequestPaymentChange(ctx context.Context, gate Gate, orders agentcore.OrderService, orderID, method string) (map[string]any, error) {
	if orderID == "" || method == "" {
		return map[string]any{"status": "error", "message": "orderId と paymentMethod が要る"}, nil
	}
	args := map[string]any{"orderId": orderID, "paymentMethod": method}
	d, err := gate.Authorize(ctx, ToolUpdatePaymentMethod, args)
	if err != nil {
		return nil, err
	}
	switch d.Outcome {
	case approval.Allow:
		order, err := orders.UpdatePaymentMethod(ctx, orderID, method)
		if err != nil {
			return nil, err
		}
		return map[string]any{"status": "done", "order": order}, nil
	case approval.Deny:
		return map[string]any{"status": "forbidden", "message": "この操作は承認があっても実行できない"}, nil
	case approval.NeedsApproval:
	}
	return map[string]any{
		"status":     StatusPending,
		"request_id": d.Request.ID,
		"risk":       d.Risk.String(),
		"input":      args,
		"message":    "承認者の承認を待っている。承認されたら依頼者が実行する",
	}, nil
}

// StatusPending はツールの結果で承認待ちを表す。
const StatusPending = "pending_approval"

// PendingFromResult はツールの結果が承認待ちなら、AskResult に載せる承認待ちに写す。
// ADK と Genkit のどちらも、ツールの結果からこれで拾う。
func PendingFromResult(tool string, out map[string]any) (agentcore.PendingToolCall, bool) {
	if out["status"] != StatusPending {
		return agentcore.PendingToolCall{}, false
	}
	id, _ := out["request_id"].(string)
	input, _ := out["input"].(map[string]any)
	return agentcore.PendingToolCall{ID: id, Name: tool, Input: input}, id != ""
}

// Executor は承認済みの依頼を、承認された引数で実行する。agentcore.ToolExecutor を満たす。
type Executor struct {
	Gate   Gate
	Orders agentcore.OrderService
}

var _ agentcore.ToolExecutor = Executor{}

func (e Executor) Execute(ctx context.Context, id string) (map[string]any, error) {
	r, err := e.Gate.Take(ctx, id)
	if err != nil {
		if liberrors.Convert(err).Code == liberrors.CodeNotFound {
			return nil, errors.Join(agentcore.ErrPendingNotFound, err)
		}
		return nil, err
	}
	switch r.Tool {
	case ToolUpdatePaymentMethod:
		orderID, _ := r.Args["orderId"].(string)
		method, _ := r.Args["paymentMethod"].(string)
		order, err := e.Orders.UpdatePaymentMethod(ctx, orderID, method)
		if err != nil {
			return nil, err
		}
		return map[string]any{"order": order}, nil
	}
	return nil, fmt.Errorf("action: 実行の手順が無いツール %q", r.Tool)
}
