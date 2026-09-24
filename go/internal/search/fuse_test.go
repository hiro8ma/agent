package search_test

import (
	"cmp"
	"context"
	"fmt"
	"math"
	"slices"
	"testing"

	"github.com/hiro8ma/agent/go/internal/search"
)

func TestFuseRRF(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		lists [][]string
		want  []string
	}{
		"両方で 2 位の文書が片方でだけ 1 位の文書より上": {
			lists: [][]string{{"a", "c"}, {"b", "c"}},
			want:  []string{"c", "a", "b"},
		},
		"同点なら先に現れた順": {
			lists: [][]string{{"a"}, {"b"}},
			want:  []string{"a", "b"},
		},
		"同じリストで重複したキーは上の順位だけを数える": {
			lists: [][]string{{"a", "a", "b"}, {"b"}},
			want:  []string{"b", "a"},
		},
		"リストが無ければ空": {
			lists: nil,
			want:  []string{},
		},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			fused := search.FuseRRF(search.DefaultRRFK, tc.lists...)
			got := make([]string, len(fused))
			for i, f := range fused {
				got[i] = f.Key
			}
			if !slices.Equal(got, tc.want) {
				t.Fatalf("keys = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestFuseRRFScore(t *testing.T) {
	t.Parallel()
	fused := search.FuseRRF(search.DefaultRRFK, []string{"a", "c"}, []string{"b", "c"})
	want := 1.0/62 + 1.0/62
	if fused[0].Key != "c" || math.Abs(fused[0].Score-want) > 1e-12 {
		t.Fatalf("top = %+v, want {c %v}", fused[0], want)
	}
}

type fakeEmbedder map[string][]float32

func (e fakeEmbedder) Embed(_ context.Context, text string) ([]float32, error) {
	v, ok := e[text]
	if !ok {
		return nil, fmt.Errorf("no vector for %q", text)
	}
	return v, nil
}

// vectorIndex は事前に与えたベクトルとのコサイン類似度で全件を並べる。
type vectorIndex struct {
	docs     []search.Doc
	vecs     [][]float32
	embedder fakeEmbedder
}

func (v vectorIndex) Search(ctx context.Context, query string, limit int) ([]search.Doc, error) {
	q, err := v.embedder.Embed(ctx, query)
	if err != nil {
		return nil, err
	}
	type scored struct {
		doc search.Doc
		sim float64
	}
	all := make([]scored, len(v.docs))
	for i, d := range v.docs {
		all[i] = scored{doc: d, sim: cosine(q, v.vecs[i])}
	}
	slices.SortStableFunc(all, func(a, b scored) int { return cmp.Compare(b.sim, a.sim) })
	out := make([]search.Doc, 0, limit)
	for _, s := range all[:min(limit, len(all))] {
		out = append(out, s.doc)
	}
	return out, nil
}

func cosine(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / math.Sqrt(na*nb)
}

func TestHybridLiftsDocRankedHighByBoth(t *testing.T) {
	t.Parallel()
	ds := []search.Doc{
		{Title: "keyword", Content: "カレーのレシピ カレー カレー"},
		{Title: "semantic", Content: "スパイスで煮込む料理"},
		{Title: "both", Content: "カレーと煮込み料理"},
		{Title: "other", Content: "寿司"},
	}
	vec := vectorIndex{
		docs:     ds,
		vecs:     [][]float32{{0, 1}, {1, 0.1}, {1, 0.5}, {0.2, 1}},
		embedder: fakeEmbedder{"カレー": {1, 0}},
	}
	bm25 := search.New(ds)
	testCases := map[string]struct {
		r       search.Retriever
		wantTop []string
	}{
		"BM25 だけなら語を多く含む文書が 1 位": {
			r:       bm25,
			wantTop: []string{"keyword", "both"},
		},
		"ベクトルだけなら意味の近い文書が 1 位": {
			r:       vec,
			wantTop: []string{"semantic", "both"},
		},
		"RRF で統合すると両方で 2 位の文書が 1 位": {
			r:       search.Hybrid{Retrievers: []search.Retriever{bm25, vec}, Depth: 2},
			wantTop: []string{"both", "keyword"},
		},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			got, err := tc.r.Search(t.Context(), "カレー", 2)
			if err != nil {
				t.Fatalf("Search: %v", err)
			}
			gotTitles := make([]string, len(got))
			for i, d := range got {
				gotTitles[i] = d.Title
			}
			if !slices.Equal(gotTitles, tc.wantTop) {
				t.Fatalf("titles = %v, want %v", gotTitles, tc.wantTop)
			}
		})
	}
}

func TestHybridReturnsRetrieverError(t *testing.T) {
	t.Parallel()
	h := search.Hybrid{Retrievers: []search.Retriever{search.New(nil), vectorIndex{embedder: fakeEmbedder{}}}}
	if _, err := h.Search(t.Context(), "カレー", 2); err == nil {
		t.Fatal("err = nil, want error")
	}
}
