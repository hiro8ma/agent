package search_test

import (
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/hiro8ma/agent/go/internal/search"
)

func hitIDs(hits []search.VectorHit) []int {
	ids := make([]int, len(hits))
	for i, h := range hits {
		ids[i] = h.ID
	}
	return ids
}

func docIDs(docs []search.Doc) []string {
	ids := make([]string, len(docs))
	for i, d := range docs {
		ids[i] = d.ID
	}
	return ids
}

func TestFlatSearchVector(t *testing.T) {
	t.Parallel()
	vecs := [][]float32{{1, 0}, {0, 1}, {3, 1}, {-1, 0}}
	f, err := search.NewFlat(make([]search.Doc, len(vecs)), vecs, nil)
	if err != nil {
		t.Fatalf("NewFlat: %v", err)
	}
	testCases := map[string]struct {
		query []float32
		limit int
		want  []int
	}{
		"長さに関係なく向きの近い順": {
			query: []float32{10, 0},
			limit: 3,
			want:  []int{0, 2, 1},
		},
		"limit が負ならすべて": {
			query: []float32{0, 1},
			limit: -1,
			want:  []int{1, 2, 0, 3},
		},
		"limit が文書数を超えてもすべて": {
			query: []float32{-1, 0},
			limit: 10,
			want:  []int{3, 1, 2, 0},
		},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			hits, err := f.SearchVector(tc.query, tc.limit)
			if err != nil {
				t.Fatalf("SearchVector: %v", err)
			}
			if got := hitIDs(hits); !slices.Equal(got, tc.want) {
				t.Fatalf("ids = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestVectorIndexRejectsDimensionMismatch(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		build func() error
	}{
		"文書のベクトルの次元がそろわない": {
			build: func() error {
				_, err := search.NewFlat(make([]search.Doc, 2), [][]float32{{1, 0}, {1}}, nil)
				return err
			},
		},
		"文書とベクトルの数が違う": {
			build: func() error {
				_, err := search.NewIVF(make([]search.Doc, 1), [][]float32{{1, 0}, {0, 1}}, nil, search.IVFConfig{})
				return err
			},
		},
		"クエリの次元が違う": {
			build: func() error {
				f, err := search.NewFlat(make([]search.Doc, 1), [][]float32{{1, 0}}, nil)
				if err != nil {
					return nil
				}
				_, err = f.SearchVector([]float32{1, 0, 0}, 1)
				return err
			},
		},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			if err := tc.build(); err == nil {
				t.Fatal("err = nil, want error")
			}
		})
	}
}

// threeClusters は向きのはっきり違う 3 つの塊を作る。0〜4 は x 軸、5〜9 は y 軸、10〜14 は z 軸の近く。
func threeClusters() [][]float32 {
	r := rand.New(rand.NewPCG(3, 3))
	var vecs [][]float32
	for axis := range 3 {
		for range 5 {
			v := []float32{0.05 * r.Float32(), 0.05 * r.Float32(), 0.05 * r.Float32()}
			v[axis] = 1
			vecs = append(vecs, v)
		}
	}
	return vecs
}

func TestIVFSearchesOnlyProbedLists(t *testing.T) {
	t.Parallel()
	vecs := threeClusters()
	ivf, err := search.NewIVF(make([]search.Doc, len(vecs)), vecs, nil, search.IVFConfig{NList: 3})
	if err != nil {
		t.Fatalf("NewIVF: %v", err)
	}
	if got := ivf.ListSizes(); !slices.Equal(got, []int{5, 5, 5}) {
		t.Fatalf("list sizes = %v, want [5 5 5]", got)
	}
	query := []float32{1, 0.4, 0}
	testCases := map[string]struct {
		nprobe   int
		wantHits int
		wantIn   []int
	}{
		"nprobe=1 なら最も近い塊の 5 件だけを返す": {
			nprobe:   1,
			wantHits: 5,
			wantIn:   []int{0, 1, 2, 3, 4},
		},
		"nprobe=2 なら次に近い塊も比べる": {
			nprobe:   2,
			wantHits: 10,
			wantIn:   []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
		},
		"nprobe が nlist を超えたらすべての塊を比べる": {
			nprobe:   10,
			wantHits: 15,
		},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			hits, err := ivf.WithNProbe(tc.nprobe).SearchVector(query, -1)
			if err != nil {
				t.Fatalf("SearchVector: %v", err)
			}
			if len(hits) != tc.wantHits {
				t.Fatalf("hits = %d, want %d", len(hits), tc.wantHits)
			}
			if tc.wantIn == nil {
				return
			}
			got := hitIDs(hits)
			slices.Sort(got)
			if !slices.Equal(got, tc.wantIn) {
				t.Fatalf("ids = %v, want %v", got, tc.wantIn)
			}
		})
	}
}

func TestIVFMatchesFlatWhenAllListsProbed(t *testing.T) {
	t.Parallel()
	vecs := clusteredVectors(2_000, 32, 20, 11)
	flat, err := search.NewFlat(make([]search.Doc, len(vecs)), vecs, nil)
	if err != nil {
		t.Fatalf("NewFlat: %v", err)
	}
	ivf, err := search.NewIVF(make([]search.Doc, len(vecs)), vecs, nil, search.IVFConfig{NList: 16})
	if err != nil {
		t.Fatalf("NewIVF: %v", err)
	}
	queries := clusteredVectors(20, 32, 20, 12)
	testCases := map[string]struct {
		nprobe     int
		wantRecall float64
		exact      bool
	}{
		"すべての塊を比べれば全件比較と同じ":   {nprobe: 16, wantRecall: 1, exact: true},
		"塊の構造があれば 4 つで 9 割以上": {nprobe: 4, wantRecall: 0.9},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			got := meanRecall(t, flat, ivf.WithNProbe(tc.nprobe), queries, 10)
			if got < tc.wantRecall || tc.exact && got != tc.wantRecall {
				t.Fatalf("recall@10 = %.3f, want %.3f (exact=%v)", got, tc.wantRecall, tc.exact)
			}
		})
	}
}

func TestIVFIsDeterministic(t *testing.T) {
	t.Parallel()
	vecs := clusteredVectors(1_000, 16, 10, 5)
	build := func() []int {
		ivf, err := search.NewIVF(make([]search.Doc, len(vecs)), vecs, nil, search.IVFConfig{NList: 12})
		if err != nil {
			t.Fatalf("NewIVF: %v", err)
		}
		return ivf.ListSizes()
	}
	if a, b := build(), build(); !slices.Equal(a, b) {
		t.Fatalf("list sizes differ between builds: %v vs %v", a, b)
	}
}

func TestIVFEmpty(t *testing.T) {
	t.Parallel()
	ivf, err := search.NewIVF(nil, nil, nil, search.IVFConfig{})
	if err != nil {
		t.Fatalf("NewIVF: %v", err)
	}
	hits, err := ivf.SearchVector([]float32{1, 0}, 10)
	if err != nil || len(hits) != 0 {
		t.Fatalf("hits = %v, err = %v, want none", hits, err)
	}
}
