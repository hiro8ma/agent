package search

import (
	"cmp"
	"context"
	"slices"
)

// DefaultRRFK は RRF の k。Cormack らの論文で使われた値で、上位の順位差を効きすぎないようにならす。
const DefaultRRFK = 60

type Fused[K comparable] struct {
	Key   K
	Score float64
}

// FuseRRF は複数の順位リストを Reciprocal Rank Fusion で 1 つにする。点数の尺度が違う方式でも順位だけで合わせられる。
// 同じリストに同じキーが複数あれば上の順位だけを数える。同点ならリストに先に現れた順に並べる。
func FuseRRF[K comparable](k float64, lists ...[]K) []Fused[K] {
	pos := make(map[K]int)
	var out []Fused[K]
	for _, list := range lists {
		seen := make(map[K]struct{}, len(list))
		for rank, key := range list {
			if _, ok := seen[key]; ok {
				continue
			}
			seen[key] = struct{}{}
			i, ok := pos[key]
			if !ok {
				i = len(out)
				pos[key] = i
				out = append(out, Fused[K]{Key: key})
			}
			out[i].Score += 1 / (k + float64(rank+1))
		}
	}
	slices.SortStableFunc(out, func(a, b Fused[K]) int { return cmp.Compare(b.Score, a.Score) })
	return out
}

// Retriever は候補を順位つきで返すマッチングの方式。Index や、埋め込みで引くベクトル検索が満たす。
type Retriever interface {
	Search(ctx context.Context, query string, limit int) ([]Doc, error)
}

// Hybrid は各 Retriever から Depth 件ずつ候補を取り、RRF で統合する。Depth が 0 以下なら limit 件、K が 0 以下なら DefaultRRFK を使う。
type Hybrid struct {
	Retrievers []Retriever
	Depth      int
	K          float64
}

func (h Hybrid) Search(ctx context.Context, query string, limit int) ([]Doc, error) {
	depth := h.Depth
	if depth <= 0 {
		depth = limit
	}
	k := h.K
	if k <= 0 {
		k = DefaultRRFK
	}
	lists := make([][]Doc, 0, len(h.Retrievers))
	for _, r := range h.Retrievers {
		docs, err := r.Search(ctx, query, depth)
		if err != nil {
			return nil, err
		}
		lists = append(lists, docs)
	}
	fused := FuseRRF(k, lists...)
	if limit >= 0 && len(fused) > limit {
		fused = fused[:limit]
	}
	out := make([]Doc, len(fused))
	for i, f := range fused {
		out[i] = f.Key
	}
	return out, nil
}
