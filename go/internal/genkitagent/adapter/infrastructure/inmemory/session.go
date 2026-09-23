package inmemory

import (
	"context"
	"sync"

	"github.com/hiro8ma/agent/go/internal/genkitagent/domain/model"
	"github.com/hiro8ma/agent/go/internal/genkitagent/domain/repository"
)

type Sessions struct {
	mu       sync.Mutex
	sessions map[string][]model.Message
}

var _ repository.Session = (*Sessions)(nil)

func NewSessions() *Sessions {
	return &Sessions{sessions: map[string][]model.Message{}}
}

func (s *Sessions) ListMessages(_ context.Context, sessionID string) ([]model.Message, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	history := s.sessions[sessionID]
	out := make([]model.Message, len(history))
	copy(out, history)
	return out, nil
}

func (s *Sessions) CreateMessages(_ context.Context, sessionID string, messages []model.Message) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.sessions[sessionID] = append(s.sessions[sessionID], messages...)
	return nil
}
