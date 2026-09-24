package search_test

import (
	"fmt"
	"math"
	"slices"
	"testing"

	"github.com/hiro8ma/agent/go/internal/search"
)

var _ search.Retriever = (*search.Sharded)(nil)

func withIDs(ds []search.Doc) []search.Doc {
	out := slices.Clone(ds)
	for i := range out {
		out[i].ID = fmt.Sprintf("id-%d", i)
	}
	return out
}

func TestShardedDFSMatchesSingleIndex(t *testing.T) {
	t.Parallel()
	ds := withIDs(randomDocs(2_000))
	testCases := map[string]struct {
		shards    int
		opts      []search.Option
		query     search.Query
		wantEmpty bool
	}{
		"1 シャード":      {shards: 1, query: search.Query{Text: benchQuery}},
		"2 シャード":      {shards: 2, query: search.Query{Text: benchQuery}},
		"5 シャード":      {shards: 5, query: search.Query{Text: "w00001 w00300 w01234"}},
		"20 シャード":     {shards: 20, query: search.Query{Text: benchQuery}},
		"BM25F の重みつき": {shards: 5, opts: []search.Option{search.WithFieldWeights(2, 1)}, query: search.Query{Text: "doc1 w00050"}},
		"本文だけを検索":     {shards: 5, query: search.Query{Text: "doc1 w00050", Fields: []search.Field{search.FieldContent}}},
		"同義語で広げたクエリ":  {shards: 5, query: search.Query{Text: "w00050", Synonyms: map[string]string{"w00050": "w00051"}}},
		"フレーズ":        {shards: 5, query: search.Query{Text: "w00001 w00002", Phrase: true}},
		"どのシャードにも無い語だけのクエリ": {shards: 5, query: search.Query{Text: "zzz"}, wantEmpty: true},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			want := search.New(ds, tc.opts...).RankQuery(tc.query, -1)
			if (len(want.Hits) == 0) != tc.wantEmpty {
				t.Fatalf("single index hits = %d, wantEmpty %v", len(want.Hits), tc.wantEmpty)
			}
			sh := search.NewSharded(ds, tc.shards, nil, tc.opts...)
			sh.Type = search.DFSQueryThenFetch
			got := sh.RankQuery(tc.query, -1)
			if len(got.Hits) != len(want.Hits) {
				t.Fatalf("len = %d, want %d", len(got.Hits), len(want.Hits))
			}
			for i := range want.Hits {
				g, w := got.Hits[i], want.Hits[i]
				if g.Doc.ID != w.Doc.ID || math.Abs(g.Score-w.Score) > 1e-9 {
					t.Fatalf("hit %d = {%s %v}, want {%s %v}", i, g.Doc.ID, g.Score, w.Doc.ID, w.Score)
				}
			}
		})
	}
}

// skewedDocs は "go" を 5 件すべてが含む前半と、1 件だけが含む後半に分かれる。前半をシャード 0、後半をシャード 1 に置く。
func skewedDocs() ([]search.Doc, search.Router) {
	ds := withIDs(docs(
		"go go java",
		"go x y",
		"go x y",
		"go x y",
		"go x y",
		"go x rust",
		"x y z",
		"x y z",
		"x y z",
		"x y z",
	))
	return ds, func(d search.Doc) int {
		var i int
		_, _ = fmt.Sscanf(d.ID, "id-%d", &i)
		return i / 5
	}
}

func TestShardedQueryThenFetchDiverges(t *testing.T) {
	t.Parallel()
	ds, router := skewedDocs()
	testCases := map[string]struct {
		typ  search.SearchType
		want []string
	}{
		"シャードの中の統計では go が珍しいシャードの文書が上がる": {typ: search.QueryThenFetch, want: []string{"d5", "d0", "d1"}},
		"全体の統計では go を 2 回含む文書が上":         {typ: search.DFSQueryThenFetch, want: []string{"d0", "d1", "d2"}},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			sh := search.NewSharded(ds, 2, router)
			sh.Type = tc.typ
			if got := sh.ShardDocs(); !slices.Equal(got, []int{5, 5}) {
				t.Fatalf("ShardDocs = %v, want [5 5]", got)
			}
			if got := titles(sh.Rank("go", 3).Hits); !slices.Equal(got, tc.want) {
				t.Fatalf("titles = %v, want %v", got, tc.want)
			}
		})
	}
	if got := titles(search.New(ds).Rank("go", 3).Hits); !slices.Equal(got, []string{"d0", "d1", "d2"}) {
		t.Fatalf("single index titles = %v", got)
	}
}

func TestShardedInHybrid(t *testing.T) {
	t.Parallel()
	ds, router := skewedDocs()
	sh := search.NewSharded(ds, 2, router)
	sh.Type = search.DFSQueryThenFetch
	got, err := search.Hybrid{Retrievers: []search.Retriever{sh}}.Search(t.Context(), "go", 2)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("len = %d, want 2", len(got))
	}
	if ids := []string{got[0].Title, got[1].Title}; !slices.Equal(ids, []string{"d0", "d1"}) {
		t.Fatalf("titles = %v, want [d0 d1]", ids)
	}
}
