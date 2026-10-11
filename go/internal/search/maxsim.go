package search

import (
	"context"
	"math"
	"slices"
)

// TokenEmbedder はトークンごとのベクトルを返す。ColBERT のように文書を1本ではなくトークンの数だけのベクトルで持つ方式に使う。
type TokenEmbedder interface {
	EmbedTokens(ctx context.Context, text string) ([][]float32, error)
}

// MaxSim は ColBERT の遅延相互作用（late interaction）の点数。クエリのトークンごとに文書のトークンとの最大のコサイン類似度を取り、足し合わせる。
// 文書のトークンのベクトルは前もって求めておけるので、交差エンコーダと違ってクエリごとに文書を推論し直さない。
// 長さ1でないベクトルは写しを正規化して比べる。文書が空なら0。
func MaxSim(query, doc [][]float32) float32 {
	if len(doc) == 0 {
		return 0
	}
	doc = unitRows(doc)
	var sum float32
	for _, q := range unitRows(query) {
		best := float32(math.Inf(-1))
		for _, d := range doc {
			best = max(best, dot(q, d))
		}
		sum += best
	}
	return sum
}

// RankMaxSim は docs のすべてと MaxSim を求め、上位 limit 件を返す。VectorHit の ID は docs の添字、Score は MaxSim。
// ColBERT はトークンのベクトルを128次元に射影する層ごと遅延相互作用を学習する。文単位の埋め込みとして学習したモデル（Ruri など）のトークンのベクトルで求める MaxSim は、比較のための近似になる。
func RankMaxSim(query [][]float32, docs [][][]float32, limit int) []VectorHit {
	query = unitRows(query)
	top := newTopK(resultSize(limit, len(docs)))
	for i, d := range docs {
		top.push(VectorHit{ID: i, Score: MaxSim(query, d)})
	}
	return top.sorted()
}

// DropEnds は先頭と末尾のトークンを除いた部分を返す。トークナイザが前後に足す特殊トークン（Ruri v3 なら <s> と </s>）を MaxSim から外すときに使う。2トークン以下なら空。
func DropEnds(vecs [][]float32) [][]float32 {
	if len(vecs) <= 2 {
		return nil
	}
	return vecs[1 : len(vecs)-1]
}

func unitRows(vs [][]float32) [][]float32 {
	var out [][]float32
	for i, v := range vs {
		if isUnit(v) {
			continue
		}
		if out == nil {
			out = slices.Clone(vs)
		}
		out[i] = slices.Clone(v)
		normalize(out[i])
	}
	if out == nil {
		return vs
	}
	return out
}

func isUnit(v []float32) bool {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	return math.Abs(sum-1) < 1e-4 || sum == 0
}
