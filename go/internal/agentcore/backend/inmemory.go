// Package backend はツールの接続先マイクロサービスと社内ナレッジ検索のローカル実装。
// 実運用では gRPC クライアント実装に差し替える。
package backend

import (
	"context"
	"fmt"
	"sync"

	"github.com/hiro8ma/agent/go/internal/agentcore"
)

type InMemoryOrders struct {
	mu     sync.Mutex
	orders map[string]agentcore.Order
}

var _ agentcore.OrderService = (*InMemoryOrders)(nil)

func NewInMemoryOrders() *InMemoryOrders {
	return &InMemoryOrders{
		orders: map[string]agentcore.Order{
			"ord-001": {ID: "ord-001", CustomerName: "山田太郎", PaymentMethod: "銀行振込", AmountJPY: 128000},
			"ord-002": {ID: "ord-002", CustomerName: "佐藤花子", PaymentMethod: "クレジットカード", AmountJPY: 39800},
		},
	}
}

func (s *InMemoryOrders) GetOrder(_ context.Context, orderID string) (*agentcore.Order, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	o, ok := s.orders[orderID]
	if !ok {
		return nil, fmt.Errorf("order %s not found", orderID)
	}
	return &o, nil
}

func (s *InMemoryOrders) UpdatePaymentMethod(_ context.Context, orderID, paymentMethod string) (*agentcore.Order, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	o, ok := s.orders[orderID]
	if !ok {
		return nil, fmt.Errorf("order %s not found", orderID)
	}
	o.PaymentMethod = paymentMethod
	s.orders[orderID] = o
	return &o, nil
}

type InMemoryGeo struct {
	areas map[string]string
}

var _ agentcore.GeoService = (*InMemoryGeo)(nil)

func NewInMemoryGeo() *InMemoryGeo {
	return &InMemoryGeo{
		areas: map[string]string{
			"area-13104": "東京都新宿区",
			"area-27127": "大阪市中央区",
		},
	}
}

func (s *InMemoryGeo) ResolveAreaNames(_ context.Context, areaIDs []string) (map[string]string, error) {
	names := make(map[string]string, len(areaIDs))
	for _, id := range areaIDs {
		if name, ok := s.areas[id]; ok {
			names[id] = name
		}
	}
	return names, nil
}
