package search_test

import (
	"slices"
	"testing"

	"github.com/hiro8ma/agent/go/internal/search"
	"github.com/hiro8ma/agent/go/internal/search/searchtest"
)

// TestTutorialRerank は教材の例で、BM25 で候補を絞ってからベクトルで並べ直した上位 3 件を、BM25 / ベクトル / RRF と比べる。
func TestTutorialRerank(t *testing.T) {
	t.Parallel()
	fx, err := searchtest.Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	vec, err := search.NewFlat(searchtest.TutorialDocs, fx.DocVectors(), fx.Embedder())
	if err != nil {
		t.Fatalf("NewFlat: %v", err)
	}
	bm25 := search.New(searchtest.TutorialDocs)
	rerank := func(cfg search.RerankConfig) search.Retriever {
		r, err := search.NewRerank(bm25, vec, cfg)
		if err != nil {
			t.Fatalf("NewRerank: %v", err)
		}
		return r
	}
	retrievers := map[string]search.Retriever{
		"bm25":         bm25,
		"vec":          vec,
		"rrf":          search.Hybrid{Retrievers: []search.Retriever{bm25, vec}},
		"rerank":       rerank(search.RerankConfig{}),
		"rerank-and":   rerank(search.RerankConfig{Operator: search.OperatorAnd}),
		"rerank-top-1": rerank(search.RerankConfig{Depth: 1}),
		"rerank-top-2": rerank(search.RerankConfig{Depth: 2}),
	}
	testCases := map[string]struct {
		query     string
		retriever string
		want      []string
	}{
		"インド料理のお店は BM25 の 2 件だけを並べ直し、ベクトルで 2 位のキーマカレーは出ない": {query: "インド料理のお店", retriever: "rerank", want: []string{"d2", "d6"}},
		"インド料理のお店は AND で絞るとインの bigram だけの転置インデックスが落ちる":     {query: "インド料理のお店", retriever: "rerank-and", want: []string{"d2"}},
		"カレーは BM25 の 3 件をベクトルの順に並べ直す":                      {query: "カレー", retriever: "rerank", want: []string{"d1", "d3", "d7"}},
		"カレーは BM25 の 1 位だけに絞ると 1 件になる":                     {query: "カレー", retriever: "rerank-top-1", want: []string{"d3"}},
		"カレーは BM25 の上位 2 件に絞ると BM25 で 3 位のカレーライスが出ない":      {query: "カレー", retriever: "rerank-top-2", want: []string{"d3", "d7"}},
		"ねこは BM25 が 0 件なので並べ直しても空":                         {query: "ねこ", retriever: "rerank", want: []string{}},
		"ねこは AND でも空": {query: "ねこ", retriever: "rerank-and", want: []string{}},
		"転置インデックスは BM25 の 2 件をベクトルでも同じ順に並べる":   {query: "転置インデックス", retriever: "rerank", want: []string{"d6", "d2"}},
		"転置インデックスは AND で絞ると 1 件になる":            {query: "転置インデックス", retriever: "rerank-and", want: []string{"d6"}},
		"比較のため BM25 のカレーは短い文書から並ぶ":             {query: "カレー", retriever: "bm25", want: []string{"d3", "d7", "d1"}},
		"比較のためベクトルのねこは猫の 2 件を返す":               {query: "ねこ", retriever: "vec", want: []string{"d4", "d5", "d1"}},
		"比較のため RRF のインド料理のお店は転置インデックスを 2 位にする": {query: "インド料理のお店", retriever: "rrf", want: []string{"d2", "d6", "d7"}},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			docs, err := retrievers[tc.retriever].Search(t.Context(), tc.query, 3)
			if err != nil {
				t.Fatalf("Search: %v", err)
			}
			if got := docIDs(docs); !slices.Equal(got, tc.want) {
				t.Fatalf("ids = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestRerankRankVector(t *testing.T) {
	t.Parallel()
	ds := []search.Doc{
		{ID: "a", Content: "go go go"},
		{ID: "b", Content: "go"},
		{ID: "c", Content: "rust"},
	}
	vecs := [][]float32{{0, 1}, {1, 0}, {1, 0.01}}
	vec, err := search.NewFlat(ds, vecs, nil)
	if err != nil {
		t.Fatalf("NewFlat: %v", err)
	}
	r, err := search.NewRerank(search.New(ds), vec, search.RerankConfig{})
	if err != nil {
		t.Fatalf("NewRerank: %v", err)
	}
	testCases := map[string]struct {
		query string
		q     []float32
		want  []int
	}{
		"BM25 の候補をベクトルの近い順に並べ替える":       {query: "go", q: []float32{1, 0}, want: []int{1, 0}},
		"ベクトルが最も近くても BM25 の候補に無い文書は出ない": {query: "go", q: []float32{1, 0.01}, want: []int{1, 0}},
		"BM25 が 0 件なら空": {query: "python", q: []float32{1, 0}, want: []int{}},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			hits, err := r.RankVector(tc.query, tc.q, 10)
			if err != nil {
				t.Fatalf("RankVector: %v", err)
			}
			got := make([]int, len(hits))
			for i, h := range hits {
				got[i] = h.ID
			}
			if !slices.Equal(got, tc.want) {
				t.Fatalf("ids = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestNewRerankRejectsMismatchedDocs(t *testing.T) {
	t.Parallel()
	vec, err := search.NewFlat([]search.Doc{{ID: "a"}}, [][]float32{{1}}, nil)
	if err != nil {
		t.Fatalf("NewFlat: %v", err)
	}
	if _, err := search.NewRerank(search.New(nil), vec, search.RerankConfig{}); err == nil {
		t.Fatal("err = nil, want error")
	}
}
