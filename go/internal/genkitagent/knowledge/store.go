// Package knowledge はデータストア（社内ナレッジ検索）のローカル実装。
// 実運用では Vertex AI Search / RAG Engine / pgvector に差し替える。
package knowledge

import (
	"context"
	"sort"
	"strings"

	"github.com/hiro8ma/agent/go/internal/genkitagent/agent"
)

type InMemory struct {
	docs []agent.KnowledgeDoc
}

var _ agent.KnowledgeSearcher = (*InMemory)(nil)

func NewInMemory() *InMemory {
	return &InMemory{
		docs: []agent.KnowledgeDoc{
			{Title: "経費精算の締め日", Content: "経費精算の申請締め日は毎月25日。締め日を過ぎた申請は翌月精算になる。承認者は所属部署の部長。"},
			{Title: "リモートワーク規程", Content: "リモートワークは週3日まで。事前申請は不要だが、チームのコアタイム 11:00-15:00 は接続必須。"},
			{Title: "注文キャンセルポリシー", Content: "出荷前の注文はキャンセル可能。出荷後は返品扱いとなり、返送料は顧客負担。支払い方法の変更は出荷前のみ可能。"},
			{Title: "技術選定ガイドライン", Content: "新規サービスのバックエンドは Go を第一候補とする。フロントエンドは Next.js。インフラは Cloud Run を標準とする。"},
		},
	}
}

// Search は単純なキーワード一致で検索する。スコアは一致した語数。
func (s *InMemory) Search(_ context.Context, query string, limit int) ([]agent.KnowledgeDoc, error) {
	terms := strings.Fields(query)
	type scored struct {
		doc   agent.KnowledgeDoc
		score int
	}
	var results []scored
	for _, d := range s.docs {
		text := d.Title + " " + d.Content
		score := 0
		for _, t := range terms {
			if strings.Contains(text, t) {
				score++
			}
		}
		if score > 0 {
			results = append(results, scored{doc: d, score: score})
		}
	}
	sort.SliceStable(results, func(i, j int) bool { return results[i].score > results[j].score })
	if len(results) > limit {
		results = results[:limit]
	}
	docs := make([]agent.KnowledgeDoc, len(results))
	for i, r := range results {
		docs[i] = r.doc
	}
	return docs, nil
}
