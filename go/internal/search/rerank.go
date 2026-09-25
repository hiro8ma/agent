package search

import (
	"context"
	"fmt"
)

// RerankConfig の Depth は BM25 から取る候補の数で、0 以下ならマッチングで残った候補をすべて並べ直す。Operator は BM25 のマッチングの組み合わせ方。
type RerankConfig struct {
	Depth    int
	Operator Operator
}

// Rerank は索引語でマッチングして BM25 の上位 Depth 件に絞り、その候補だけをクエリのベクトルとのコサイン類似度で並べ直す。
// 候補の中は全件と比べる。BM25 の候補に入らない文書は、ベクトルが近くても結果に出ない。
type Rerank struct {
	ix  *Index
	vec *Flat
	cfg RerankConfig
}

var _ Retriever = (*Rerank)(nil)

// NewRerank の vec は ix と同じ文書を同じ順に持つ Flat にする。ベクトルと埋め込みのモデルは vec のものを使う。
func NewRerank(ix *Index, vec *Flat, cfg RerankConfig) (*Rerank, error) {
	if len(ix.docs) != vec.vecs.len() {
		return nil, fmt.Errorf("search: index has %d docs but %d vectors", len(ix.docs), vec.vecs.len())
	}
	return &Rerank{ix: ix, vec: vec, cfg: cfg}, nil
}

// RankVector はクエリの文字列で候補を絞り、q との類似度で上位 limit 件を返す。VectorHit の ID は索引に渡した文書の添字。
func (r *Rerank) RankVector(query string, q []float32, limit int) ([]VectorHit, error) {
	q, err := r.vec.vecs.query(q)
	if err != nil {
		return nil, err
	}
	depth := r.cfg.Depth
	if depth <= 0 {
		depth = -1
	}
	_, ids := r.ix.rankWith(Query{Text: query, Operator: r.cfg.Operator}, depth, nil)
	top := newTopK(resultSize(limit, len(ids)))
	for _, id := range ids {
		top.push(VectorHit{ID: id, Score: dot(q, r.vec.vecs.row(id))})
	}
	return top.sorted(), nil
}

func (r *Rerank) Search(ctx context.Context, query string, limit int) ([]Doc, error) {
	q, err := embedQuery(ctx, r.vec.embedder, query)
	if err != nil {
		return nil, err
	}
	hits, err := r.RankVector(query, q, limit)
	if err != nil {
		return nil, err
	}
	return hitsToDocs(r.ix.docs, hits), nil
}
