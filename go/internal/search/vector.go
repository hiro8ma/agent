package search

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"math"
	"slices"
)

// Embedder は文字列をベクトルにする。文書とクエリで同じモデルを使わないと、次元が合っても類似度に意味が無くなる。
type Embedder interface {
	Embed(ctx context.Context, text string) ([]float32, error)
}

// VectorHit の ID は索引に渡した文書の添字。Score はコサイン類似度。
type VectorHit struct {
	ID    int
	Score float32
}

// vectorSet は長さ 1 に正規化したベクトルを行ごとに詰めて持つ。内積がそのままコサイン類似度になる。
type vectorSet struct {
	dim  int
	data []float32
}

func newVectorSet(vecs [][]float32) (vectorSet, error) {
	if len(vecs) == 0 {
		return vectorSet{}, nil
	}
	dim := len(vecs[0])
	if dim == 0 {
		return vectorSet{}, errors.New("search: vector has no dimensions")
	}
	s := vectorSet{dim: dim, data: make([]float32, 0, len(vecs)*dim)}
	for i, v := range vecs {
		if len(v) != dim {
			return vectorSet{}, fmt.Errorf("search: vector %d has %d dimensions, want %d", i, len(v), dim)
		}
		s.data = append(s.data, v...)
		normalize(s.data[i*dim:])
	}
	return s, nil
}

func (s vectorSet) len() int {
	if s.dim == 0 {
		return 0
	}
	return len(s.data) / s.dim
}

func (s vectorSet) row(i int) []float32 { return s.data[i*s.dim : (i+1)*s.dim] }

func (s vectorSet) query(q []float32) ([]float32, error) {
	if s.dim != 0 && len(q) != s.dim {
		return nil, fmt.Errorf("search: query has %d dimensions, want %d", len(q), s.dim)
	}
	out := slices.Clone(q)
	normalize(out)
	return out, nil
}

func normalize(v []float32) {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if sum == 0 {
		return
	}
	inv := float32(1 / math.Sqrt(sum))
	for i := range v {
		v[i] *= inv
	}
}

func dot(a, b []float32) float32 {
	b = b[:len(a)]
	var s0, s1, s2, s3 float32
	i := 0
	for ; i+4 <= len(a); i += 4 {
		s0 += a[i] * b[i]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
	}
	for ; i < len(a); i++ {
		s0 += a[i] * b[i]
	}
	return s0 + s1 + s2 + s3
}

func better(a, b VectorHit) bool {
	return a.Score > b.Score || (a.Score == b.Score && a.ID < b.ID)
}

// topK は上位 k 件を最小ヒープで持つ。先頭が残っている中で最も悪い候補になる。
type topK struct {
	k    int
	heap []VectorHit
}

func newTopK(k int) *topK { return &topK{k: k, heap: make([]VectorHit, 0, k)} }

func (t *topK) push(h VectorHit) {
	if t.k <= 0 {
		return
	}
	if len(t.heap) < t.k {
		t.heap = append(t.heap, h)
		for i := len(t.heap) - 1; i > 0; {
			p := (i - 1) / 2
			if !better(t.heap[p], t.heap[i]) {
				break
			}
			t.heap[p], t.heap[i] = t.heap[i], t.heap[p]
			i = p
		}
		return
	}
	if !better(h, t.heap[0]) {
		return
	}
	t.heap[0] = h
	for i := 0; ; {
		l, r, w := 2*i+1, 2*i+2, i
		if l < len(t.heap) && better(t.heap[w], t.heap[l]) {
			w = l
		}
		if r < len(t.heap) && better(t.heap[w], t.heap[r]) {
			w = r
		}
		if w == i {
			break
		}
		t.heap[i], t.heap[w] = t.heap[w], t.heap[i]
		i = w
	}
}

func (t *topK) sorted() []VectorHit {
	out := slices.Clone(t.heap)
	slices.SortFunc(out, func(a, b VectorHit) int {
		return cmp.Or(cmp.Compare(b.Score, a.Score), cmp.Compare(a.ID, b.ID))
	})
	return out
}

func resultSize(limit, n int) int {
	if limit < 0 || limit > n {
		return n
	}
	return limit
}

func embedQuery(ctx context.Context, e Embedder, query string) ([]float32, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if e == nil {
		return nil, errors.New("search: no embedder")
	}
	q, err := e.Embed(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("search: embed query: %w", err)
	}
	return q, nil
}

func hitsToDocs(docs []Doc, hits []VectorHit) []Doc {
	out := make([]Doc, len(hits))
	for i, h := range hits {
		out[i] = docs[h.ID]
	}
	return out
}

// Flat はクエリとすべての文書のコサイン類似度を求める厳密な最近傍探索。近似の索引の再現率を測る基準にもなる。
type Flat struct {
	docs     []Doc
	vecs     vectorSet
	embedder Embedder
}

var _ Retriever = (*Flat)(nil)

// NewFlat の vecs は docs と同じ順に並べる。ベクトルは正規化した写しを持つので、呼び出し側の値は変えない。
func NewFlat(docs []Doc, vecs [][]float32, e Embedder) (*Flat, error) {
	if len(docs) != len(vecs) {
		return nil, fmt.Errorf("search: %d docs but %d vectors", len(docs), len(vecs))
	}
	s, err := newVectorSet(vecs)
	if err != nil {
		return nil, err
	}
	return &Flat{docs: docs, vecs: s, embedder: e}, nil
}

func (f *Flat) SearchVector(q []float32, limit int) ([]VectorHit, error) {
	q, err := f.vecs.query(q)
	if err != nil {
		return nil, err
	}
	n := f.vecs.len()
	top := newTopK(resultSize(limit, n))
	for i := range n {
		top.push(VectorHit{ID: i, Score: dot(q, f.vecs.row(i))})
	}
	return top.sorted(), nil
}

func (f *Flat) Search(ctx context.Context, query string, limit int) ([]Doc, error) {
	q, err := embedQuery(ctx, f.embedder, query)
	if err != nil {
		return nil, err
	}
	hits, err := f.SearchVector(q, limit)
	if err != nil {
		return nil, err
	}
	return hitsToDocs(f.docs, hits), nil
}
