package search

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"slices"
)

// Reranker は文書とクエリを1つの系列として入れ、[CLS] から関連度を出す交差エンコーダ。文書ごとに1回推論するので、候補を絞ってから使う。
// 返す点数は docs と同じ順に並べ、大きいほど関連が強い。
type Reranker interface {
	Rerank(ctx context.Context, query string, docs []string) ([]float32, error)
}

// CrossRerankConfig の Depth は第1段から取る候補の数。limit より小さければ limit 件取る。
type CrossRerankConfig struct {
	Depth int
}

// CrossRerank は第1段の Retriever で候補を絞り、その候補だけを交差エンコーダの点数で並べ直す。
// 第1段の候補に入らない文書は、交差エンコーダが高く採点するはずでも結果に出ない。
type CrossRerank struct {
	first    Retriever
	reranker Reranker
	cfg      CrossRerankConfig
}

var _ Retriever = (*CrossRerank)(nil)

func NewCrossRerank(first Retriever, r Reranker, cfg CrossRerankConfig) (*CrossRerank, error) {
	if first == nil || r == nil {
		return nil, errors.New("search: cross rerank needs a first stage and a reranker")
	}
	return &CrossRerank{first: first, reranker: r, cfg: cfg}, nil
}

func (c *CrossRerank) Search(ctx context.Context, query string, limit int) ([]Doc, error) {
	depth := c.cfg.Depth
	if limit < 0 {
		depth = -1
	} else if depth < limit {
		depth = limit
	}
	cands, err := c.first.Search(ctx, query, depth)
	if err != nil {
		return nil, err
	}
	if len(cands) == 0 {
		return []Doc{}, nil
	}
	texts := make([]string, len(cands))
	for i, d := range cands {
		texts[i] = rerankText(d)
	}
	scores, err := c.reranker.Rerank(ctx, query, texts)
	if err != nil {
		return nil, fmt.Errorf("search: rerank: %w", err)
	}
	if len(scores) != len(cands) {
		return nil, fmt.Errorf("search: reranker returned %d scores for %d docs", len(scores), len(cands))
	}
	// 同点は第1段の順位で決める。Doc の ID は文字列で、番号の大小と辞書順が食い違う。
	order := make([]int, len(cands))
	for i := range order {
		order[i] = i
	}
	slices.SortFunc(order, func(a, b int) int {
		return cmp.Or(cmp.Compare(scores[b], scores[a]), cmp.Compare(a, b))
	})
	out := make([]Doc, resultSize(limit, len(cands)))
	for i := range out {
		out[i] = cands[order[i]]
	}
	return out, nil
}

func rerankText(d Doc) string {
	if d.Title == "" {
		return d.Content
	}
	return d.Title + "\n" + d.Content
}
