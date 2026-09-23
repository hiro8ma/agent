// Package externalservice は Genkit 版のエージェントが呼ぶ外部サービスのインターフェース。
package externalservice

import (
	"context"

	"github.com/hiro8ma/agent/go/internal/genkitagent/domain/model"
)

type OrderService interface {
	GetOrder(ctx context.Context, orderID string) (*model.Order, error)
	UpdatePaymentMethod(ctx context.Context, orderID, paymentMethod string) (*model.Order, error)
}

type GeoService interface {
	ResolveAreaNames(ctx context.Context, areaIDs []string) (map[string]string, error)
}
