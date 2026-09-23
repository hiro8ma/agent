// Package inmemory はツールの接続先と保存先のプロセス内の実装。実運用では外部のサービスに差し替える。
package inmemory

import (
	"context"
	"fmt"
	"sync"

	"github.com/hiro8ma/agent/go/internal/genkitagent/domain/externalservice"
	"github.com/hiro8ma/agent/go/internal/genkitagent/domain/model"
)

type Orders struct {
	mu     sync.Mutex
	orders map[string]model.Order
}

var _ externalservice.OrderService = (*Orders)(nil)

func NewOrders() *Orders {
	return &Orders{
		orders: map[string]model.Order{
			"ord-001": {ID: "ord-001", CustomerName: "山田太郎", PaymentMethod: "銀行振込", AmountJPY: 128000},
			"ord-002": {ID: "ord-002", CustomerName: "佐藤花子", PaymentMethod: "クレジットカード", AmountJPY: 39800},
		},
	}
}

func (s *Orders) GetOrder(_ context.Context, orderID string) (*model.Order, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	o, ok := s.orders[orderID]
	if !ok {
		return nil, fmt.Errorf("order %s not found", orderID)
	}
	return &o, nil
}

func (s *Orders) UpdatePaymentMethod(_ context.Context, orderID, paymentMethod string) (*model.Order, error) {
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

type Geo struct {
	areas map[string]string
}

var _ externalservice.GeoService = (*Geo)(nil)

func NewGeo() *Geo {
	return &Geo{
		areas: map[string]string{
			"area-13104": "東京都新宿区",
			"area-27127": "大阪市中央区",
		},
	}
}

func (s *Geo) ResolveAreaNames(_ context.Context, areaIDs []string) (map[string]string, error) {
	names := make(map[string]string, len(areaIDs))
	for _, id := range areaIDs {
		if name, ok := s.areas[id]; ok {
			names[id] = name
		}
	}
	return names, nil
}
