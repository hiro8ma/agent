package agent

import (
	"context"
	"fmt"
)

// Executor は承認済みツール呼び出しを実行する。
type Executor struct {
	orders  OrderService
	pending PendingStore
}

func NewExecutor(orders OrderService, pending PendingStore) *Executor {
	return &Executor{orders: orders, pending: pending}
}

// Execute は承認待ちを取り出して実行する。取り出しは一度きりで、二重実行はできない。
func (e *Executor) Execute(ctx context.Context, toolCallID string) (map[string]any, error) {
	p, err := e.pending.Take(ctx, toolCallID)
	if err != nil {
		return nil, err
	}
	switch p.Name {
	case updateOrderPaymentMethodTool:
		orderID, _ := p.Input["orderId"].(string)
		paymentMethod, _ := p.Input["paymentMethod"].(string)
		order, err := e.orders.UpdatePaymentMethod(ctx, orderID, paymentMethod)
		if err != nil {
			return nil, err
		}
		return map[string]any{"order": order}, nil
	default:
		return nil, fmt.Errorf("unknown pending tool %q", p.Name)
	}
}
