// Package repository は Genkit 版のエージェントが使う保存先のインターフェース。
package repository

import (
	"context"

	"github.com/hiro8ma/agent/go/internal/genkitagent/domain/model"
)

type Session interface {
	ListMessages(ctx context.Context, sessionID string) ([]model.Message, error)
	CreateMessages(ctx context.Context, sessionID string, messages []model.Message) error
}

// SessionCreator はセッション ID を発行できる保存先。Chat で session_id が空なら、これを満たす保存先に作らせる。
type SessionCreator interface {
	Session
	CreateSession(ctx context.Context, agentID string) (string, error)
}
