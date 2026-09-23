package knowledgebm25_test

import (
	"testing"

	"github.com/hiro8ma/agent/go/internal/genkitagent/adapter/infrastructure/knowledgebm25"
	"github.com/hiro8ma/agent/go/internal/genkitagent/domain/model"
)

func TestSearchReturnsMatchingDocsInScoreOrder(t *testing.T) {
	t.Parallel()
	s := knowledgebm25.New([]model.KnowledgeDoc{
		{Title: "締め日", Content: "経費精算の締め日は毎月25日"},
		{Title: "リモート", Content: "リモートワークは週3日まで"},
		{Title: "経費", Content: "経費精算 経費精算 の承認者は部長"},
	})

	got, err := s.Search(t.Context(), "経費精算", 10)
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	if len(got) != 2 || got[0].Title != "経費" || got[1].Title != "締め日" {
		t.Errorf("Search() = %+v, want [経費 締め日]", got)
	}
}
