package externalservice

import (
	"context"

	"github.com/hiro8ma/agent/go/internal/genkitagent/domain/model"
)

// ActionGate は変更系の操作の承認の窓口。Take は承認待ちが無ければ model.ErrPendingNotFound を包んで返す。
type ActionGate interface {
	Authorize(ctx context.Context, tool string, args map[string]any) (*model.ActionDecision, error)
	Take(ctx context.Context, id string) (*model.ApprovedAction, error)
}
