// Package usecase は AgentService の RPC ごとのユースケース。
package usecase

import (
	"context"
	"iter"
	"log/slog"

	"github.com/hiro8ma/agent/go/internal/genkitagent/domain/externalservice"
	"github.com/hiro8ma/agent/go/internal/genkitagent/domain/repository"
	"github.com/hiro8ma/agent/go/internal/genkitagent/domain/service"
	"github.com/hiro8ma/agent/go/internal/lib/libbudget"
)

// AgentService は AgentService の RPC と同じ名前と形のメソッドを持つ。
type AgentService interface {
	ListAgents(ctx context.Context, req *ListAgentsRequest) (*ListAgentsResponse, error)
	Chat(ctx context.Context, req *ChatRequest) iter.Seq2[*ChatResponse, error]
	ExecuteConfirmedToolCall(ctx context.Context, req *ExecuteConfirmedToolCallRequest) (*ExecuteConfirmedToolCallResponse, error)
}

type agentService struct {
	agents   *service.Registry
	sessions repository.Session
	gate     externalservice.ActionGate
	orders   externalservice.OrderService
	logger   *slog.Logger
	budget   *libbudget.Tracker
}

type Option func(*agentService)

// WithBudget を付けなければトークン予算を確かめない。
func WithBudget(b *libbudget.Tracker) Option {
	return func(s *agentService) { s.budget = b }
}

func NewAgentService(
	agents *service.Registry,
	sessions repository.Session,
	gate externalservice.ActionGate,
	orders externalservice.OrderService,
	logger *slog.Logger,
	opts ...Option,
) AgentService {
	s := &agentService{agents: agents, sessions: sessions, gate: gate, orders: orders, logger: logger}
	for _, o := range opts {
		o(s)
	}
	return s
}
