// Package knowledgebm25 は転置インデックスと BM25 の検索（internal/search）で社内ナレッジを引く。
package knowledgebm25

import (
	"context"

	"github.com/hiro8ma/agent/go/internal/genkitagent/domain/externalservice"
	"github.com/hiro8ma/agent/go/internal/genkitagent/domain/model"
	"github.com/hiro8ma/agent/go/internal/search"
)

type Searcher struct {
	index *search.Index
}

var _ externalservice.KnowledgeSearcher = (*Searcher)(nil)

func New(docs []model.KnowledgeDoc, opts ...search.Option) *Searcher {
	ds := make([]search.Doc, len(docs))
	for i, d := range docs {
		ds[i] = search.Doc{Title: d.Title, Content: d.Content}
	}
	return &Searcher{index: search.New(ds, opts...)}
}

func (s *Searcher) Search(ctx context.Context, query string, limit int) ([]model.KnowledgeDoc, error) {
	docs, err := s.index.Search(ctx, query, limit)
	if err != nil {
		return nil, err
	}
	out := make([]model.KnowledgeDoc, len(docs))
	for i, d := range docs {
		out[i] = model.KnowledgeDoc{Title: d.Title, Content: d.Content}
	}
	return out, nil
}
