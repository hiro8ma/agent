package search_test

import (
	"math"
	"slices"
	"testing"

	"github.com/hiro8ma/agent/go/internal/search"
)

func TestMaxSim(t *testing.T) {
	t.Parallel()
	s := float32(math.Sqrt(0.5))
	testCases := map[string]struct {
		query, doc [][]float32
		want       float32
	}{
		"クエリのトークンごとに最も近い文書のトークンを取って足す": {
			query: [][]float32{{1, 0}, {0, 1}},
			doc:   [][]float32{{1, 0}, {s, s}},
			want:  1 + s,
		},
		"同じトークンの文書はクエリのトークン数になる": {
			query: [][]float32{{1, 0}, {0, 1}, {s, s}},
			doc:   [][]float32{{1, 0}, {0, 1}, {s, s}},
			want:  3,
		},
		"文書の1つのトークンを複数のクエリのトークンが使ってよい": {
			query: [][]float32{{1, 0}, {s, s}},
			doc:   [][]float32{{1, 0}, {-1, 0}},
			want:  1 + s,
		},
		"長さ1でないベクトルは正規化して比べる": {
			query: [][]float32{{3, 0}},
			doc:   [][]float32{{0, 2}, {5, 5}},
			want:  s,
		},
		"文書が空なら0": {
			query: [][]float32{{1, 0}},
			doc:   nil,
			want:  0,
		},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			if got := search.MaxSim(tc.query, tc.doc); math.Abs(float64(got-tc.want)) > 1e-6 {
				t.Errorf("MaxSim = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestMaxSimKeepsInput(t *testing.T) {
	t.Parallel()
	q := [][]float32{{3, 4}}
	d := [][]float32{{0, 2}}
	search.MaxSim(q, d)
	if !slices.Equal(q[0], []float32{3, 4}) || !slices.Equal(d[0], []float32{0, 2}) {
		t.Errorf("inputs changed to %v, %v", q, d)
	}
}

func TestRankMaxSim(t *testing.T) {
	t.Parallel()
	s := float32(math.Sqrt(0.5))
	query := [][]float32{{1, 0}, {0, 1}}
	docs := [][][]float32{
		{{1, 0}},         // 1
		{{1, 0}, {0, 1}}, // 2
		{{s, s}},         // 2s
		{{0, 1}},         // 1
	}
	testCases := map[string]struct {
		limit int
		want  []int
	}{
		"MaxSim の大きい順で、同点は添字の小さい順": {limit: 4, want: []int{1, 2, 0, 3}},
		"上位 limit 件だけを返す":          {limit: 2, want: []int{1, 2}},
		"limit が負ならすべて返す":          {limit: -1, want: []int{1, 2, 0, 3}},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			hits := search.RankMaxSim(query, docs, tc.limit)
			got := make([]int, len(hits))
			for i, h := range hits {
				got[i] = h.ID
			}
			if !slices.Equal(got, tc.want) {
				t.Errorf("ids = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestDropEnds(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		in   [][]float32
		want [][]float32
	}{
		"先頭と末尾を除く":     {in: [][]float32{{1}, {2}, {3}, {4}}, want: [][]float32{{2}, {3}}},
		"3トークンなら真ん中だけ": {in: [][]float32{{1}, {2}, {3}}, want: [][]float32{{2}}},
		"2トークン以下なら空":   {in: [][]float32{{1}, {2}}, want: nil},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			got := search.DropEnds(tc.in)
			if !slices.EqualFunc(got, tc.want, slices.Equal[[]float32]) {
				t.Errorf("DropEnds = %v, want %v", got, tc.want)
			}
		})
	}
}
