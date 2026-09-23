// Package usecase は AgentService の RPC ごとのユースケース。
package usecase

import (
	"log/slog"

	"github.com/hiro8ma/agent/go/internal/genkitagent/domain/externalservice"
	"github.com/hiro8ma/agent/go/internal/genkitagent/domain/repository"
	"github.com/hiro8ma/agent/go/internal/genkitagent/domain/service"
	"github.com/hiro8ma/agent/go/internal/lib/libbudget"
)

type AgentService struct {
	agents   *service.Registry
	sessions repository.Session
	gate     externalservice.ActionGate
	orders   externalservice.OrderService
	logger   *slog.Logger
	budget   *libbudget.Tracker
}

type Option func(*AgentService)

// WithBudget を付けなければトークン予算を確かめない。
func WithBudget(b *libbudget.Tracker) Option {
	return func(s *AgentService) { s.budget = b }
}

func NewAgentService(
	agents *service.Registry,
	sessions repository.Session,
	gate externalservice.ActionGate,
	orders externalservice.OrderService,
	logger *slog.Logger,
	opts ...Option,
) *AgentService {
	s := &AgentService{agents: agents, sessions: sessions, gate: gate, orders: orders, logger: logger}
	for _, o := range opts {
		o(s)
	}
	return s
}
