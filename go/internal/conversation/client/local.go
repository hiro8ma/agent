package client

import (
	"context"
	"errors"

	"github.com/hiro8ma/agent/go/internal/agentcore"
	"github.com/hiro8ma/agent/go/internal/conversation/domain"
	"github.com/hiro8ma/agent/go/internal/conversation/usecase"
)

// Local は ConversationService を別プロセスに立てずに、同じ usecase を直接呼ぶ。
// 所有者の確認は Remote と同じく usecase で行う。
type Local struct {
	svc *usecase.Service
}

var _ agentcore.SessionCreator = (*Local)(nil)

func NewLocal(svc *usecase.Service) *Local {
	return &Local{svc: svc}
}

func (l *Local) Create(ctx context.Context, agentID string) (string, error) {
	s, err := l.svc.CreateSession(ctx, agentID)
	return s.ID, err
}

func (l *Local) Load(ctx context.Context, sessionID string) ([]agentcore.Message, error) {
	var out []agentcore.Message
	token := ""
	for {
		msgs, next, err := l.svc.ListMessages(ctx, sessionID, 0, token)
		if errors.Is(err, domain.ErrSessionNotFound) && token == "" {
			return nil, nil
		}
		if err != nil {
			return nil, err
		}
		for _, m := range msgs {
			out = append(out, agentcore.Message{Role: string(m.Role), Text: m.Text})
		}
		if next == "" {
			return out, nil
		}
		token = next
	}
}

func (l *Local) Append(ctx context.Context, sessionID string, messages ...agentcore.Message) error {
	msgs := make([]domain.Message, len(messages))
	for i, m := range messages {
		msgs[i] = domain.Message{Role: domain.Role(m.Role), Text: m.Text}
	}
	_, err := l.svc.AppendTurn(ctx, sessionID, "", msgs)
	return err
}
