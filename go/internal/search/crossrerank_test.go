package search_test

import (
	"context"
	"errors"
	"slices"
	"sync"
	"testing"

	"github.com/hiro8ma/agent/go/internal/search"
)

// fakeReranker は本文ごとに決めた点数を返し、受け取った本文を記録する。
type fakeReranker struct {
	scores map[string]float32
	err    error
	mu     sync.Mutex
	got    [][]string
}

func (f *fakeReranker) Rerank(_ context.Context, _ string, docs []string) ([]float32, error) {
	f.mu.Lock()
	f.got = append(f.got, slices.Clone(docs))
	f.mu.Unlock()
	if f.err != nil {
		return nil, f.err
	}
	out := make([]float32, len(docs))
	for i, d := range docs {
		out[i] = f.scores[d]
	}
	return out, nil
}

// fixedRetriever は limit に関係なく決めた順の文書を、limit 件までに切って返す。
type fixedRetriever []search.Doc

func (r fixedRetriever) Search(_ context.Context, _ string, limit int) ([]search.Doc, error) {
	if limit < 0 || limit > len(r) {
		return r, nil
	}
	return r[:limit], nil
}

func TestCrossRerank(t *testing.T) {
	t.Parallel()
	first := fixedRetriever{
		{ID: "a", Content: "A"},
		{ID: "b", Content: "B"},
		{ID: "c", Content: "C"},
		{ID: "d", Content: "D"},
	}
	scores := map[string]float32{"A": 0.1, "B": 0.2, "C": 0.9, "D": 0.95}
	testCases := map[string]struct {
		depth    int
		limit    int
		scores   map[string]float32
		wantIDs  []string
		wantDocs []string
	}{
		"候補を Depth 件に絞り、交差エンコーダの点数で並べ直す": {
			depth: 3, limit: 2, scores: scores,
			wantIDs: []string{"c", "b"}, wantDocs: []string{"A", "B", "C"},
		},
		"第1段の候補に入らない文書は点数が最も高くても出ない": {
			depth: 2, limit: 2, scores: scores,
			wantIDs: []string{"b", "a"}, wantDocs: []string{"A", "B"},
		},
		"limit が Depth より大きければ limit 件を候補にする": {
			depth: 1, limit: 3, scores: scores,
			wantIDs: []string{"c", "b", "a"}, wantDocs: []string{"A", "B", "C"},
		},
		"Depth が0なら limit 件を候補にする": {
			depth: 0, limit: 2, scores: scores,
			wantIDs: []string{"b", "a"}, wantDocs: []string{"A", "B"},
		},
		"同点は第1段の順を保つ": {
			depth: 4, limit: 4, scores: map[string]float32{"A": 0.5, "B": 0.5, "C": 0.5, "D": 0.5},
			wantIDs: []string{"a", "b", "c", "d"}, wantDocs: []string{"A", "B", "C", "D"},
		},
		"limit が負なら第1段のすべてを並べ直す": {
			depth: 1, limit: -1, scores: scores,
			wantIDs: []string{"d", "c", "b", "a"}, wantDocs: []string{"A", "B", "C", "D"},
		},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			fake := &fakeReranker{scores: tc.scores}
			r, err := search.NewCrossRerank(first, fake, search.CrossRerankConfig{Depth: tc.depth})
			if err != nil {
				t.Fatalf("NewCrossRerank: %v", err)
			}
			docs, err := r.Search(t.Context(), "q", tc.limit)
			if err != nil {
				t.Fatalf("Search: %v", err)
			}
			if got := docIDs(docs); !slices.Equal(got, tc.wantIDs) {
				t.Errorf("ids = %v, want %v", got, tc.wantIDs)
			}
			if len(fake.got) != 1 || !slices.Equal(fake.got[0], tc.wantDocs) {
				t.Errorf("reranker got %v, want [%v]", fake.got, tc.wantDocs)
			}
		})
	}
}

func TestCrossRerankTitle(t *testing.T) {
	t.Parallel()
	fake := &fakeReranker{}
	r, err := search.NewCrossRerank(fixedRetriever{{ID: "a", Title: "題", Content: "本文"}}, fake, search.CrossRerankConfig{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := r.Search(t.Context(), "q", 1); err != nil {
		t.Fatal(err)
	}
	if want := []string{"題\n本文"}; len(fake.got) != 1 || !slices.Equal(fake.got[0], want) {
		t.Errorf("reranker got %q, want [%q]", fake.got, want)
	}
}

func TestCrossRerankEmpty(t *testing.T) {
	t.Parallel()
	fake := &fakeReranker{}
	r, err := search.NewCrossRerank(fixedRetriever{}, fake, search.CrossRerankConfig{Depth: 10})
	if err != nil {
		t.Fatal(err)
	}
	docs, err := r.Search(t.Context(), "q", 3)
	if err != nil {
		t.Fatal(err)
	}
	if len(docs) != 0 || len(fake.got) != 0 {
		t.Errorf("docs = %v, reranker calls = %d, want none", docs, len(fake.got))
	}
}

// failingRetriever は Search で必ず err を返す。
type failingRetriever struct{ err error }

func (r failingRetriever) Search(context.Context, string, int) ([]search.Doc, error) {
	return nil, r.err
}

// shortReranker は点数を1件少なく返す。
type shortReranker struct{}

func (shortReranker) Rerank(_ context.Context, _ string, docs []string) ([]float32, error) {
	return make([]float32, len(docs)-1), nil
}

func TestCrossRerankErrors(t *testing.T) {
	t.Parallel()
	errFirst := errors.New("first stage down")
	errRerank := errors.New("reranker down")
	docs := fixedRetriever{{ID: "a", Content: "A"}, {ID: "b", Content: "B"}}
	testCases := map[string]struct {
		first    search.Retriever
		reranker search.Reranker
		want     error
	}{
		"第1段の誤りを返す":        {first: failingRetriever{err: errFirst}, reranker: &fakeReranker{}, want: errFirst},
		"交差エンコーダの誤りを包んで返す": {first: docs, reranker: &fakeReranker{err: errRerank}, want: errRerank},
		"点数の件数が合わなければ誤り":   {first: docs, reranker: shortReranker{}},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			r, err := search.NewCrossRerank(tc.first, tc.reranker, search.CrossRerankConfig{})
			if err != nil {
				t.Fatal(err)
			}
			_, err = r.Search(t.Context(), "q", 2)
			if err == nil {
				t.Fatal("err = nil, want error")
			}
			if tc.want != nil && !errors.Is(err, tc.want) {
				t.Errorf("err = %v, want %v", err, tc.want)
			}
		})
	}
}

func TestNewCrossRerankRejectsNil(t *testing.T) {
	t.Parallel()
	if _, err := search.NewCrossRerank(nil, &fakeReranker{}, search.CrossRerankConfig{}); err == nil {
		t.Error("first = nil: err = nil, want error")
	}
	if _, err := search.NewCrossRerank(fixedRetriever{}, nil, search.CrossRerankConfig{}); err == nil {
		t.Error("reranker = nil: err = nil, want error")
	}
}
