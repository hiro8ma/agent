// Package session は会話履歴と承認待ちツール呼び出しの永続化。
package session

import (
	"context"
	"sync"

	"github.com/hiro8ma/agent/go/internal/genkitagent/agent"
)

// Store はセッション単位の履歴ストア。
type Store interface {
	Load(ctx context.Context, sessionID string) ([]agent.Message, error)
	Append(ctx context.Context, sessionID string, messages ...agent.Message) error
}

type InMemory struct {
	mu       sync.Mutex
	sessions map[string][]agent.Message
}

var _ Store = (*InMemory)(nil)

func NewInMemory() *InMemory {
	return &InMemory{sessions: map[string][]agent.Message{}}
}

func (s *InMemory) Load(_ context.Context, sessionID string) ([]agent.Message, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	history := s.sessions[sessionID]
	out := make([]agent.Message, len(history))
	copy(out, history)
	return out, nil
}

func (s *InMemory) Append(_ context.Context, sessionID string, messages ...agent.Message) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.sessions[sessionID] = append(s.sessions[sessionID], messages...)
	return nil
}

// InMemoryPending は承認待ちツール呼び出しのインメモリ実装。
type InMemoryPending struct {
	mu      sync.Mutex
	pending map[string]agent.PendingToolCall
}

var _ agent.PendingStore = (*InMemoryPending)(nil)

func NewInMemoryPending() *InMemoryPending {
	return &InMemoryPending{pending: map[string]agent.PendingToolCall{}}
}

func (s *InMemoryPending) Save(_ context.Context, p agent.PendingToolCall) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.pending[p.ID] = p
	return nil
}

func (s *InMemoryPending) Take(_ context.Context, id string) (*agent.PendingToolCall, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	p, ok := s.pending[id]
	if !ok {
		return nil, agent.ErrPendingNotFound
	}
	delete(s.pending, id)
	return &p, nil
}
