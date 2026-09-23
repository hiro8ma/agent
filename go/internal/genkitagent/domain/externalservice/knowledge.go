package externalservice

import (
	"context"

	"github.com/hiro8ma/agent/go/internal/genkitagent/domain/model"
)

type KnowledgeSearcher interface {
	Search(ctx context.Context, query string, limit int) ([]model.KnowledgeDoc, error)
}
