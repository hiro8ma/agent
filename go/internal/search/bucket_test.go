package search_test

import (
	"slices"
	"testing"

	"github.com/hiro8ma/agent/go/internal/search"
)

func TestRankBucket(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		docs       []search.Doc
		query      search.Query
		wantBM25   []string
		wantBucket []string
	}{
		"words: 珍しい語に 1 つ一致する文書より、ありふれた語に 2 つ一致する文書を上にする": {
			docs: []search.Doc{
				{ID: "rare", Content: "zebra lives here"},
				{ID: "both", Content: "cat dog"},
				{ID: "c1", Content: "cat home"},
				{ID: "c2", Content: "cat food"},
				{ID: "c3", Content: "cat toy"},
				{ID: "c4", Content: "cat bed"},
				{ID: "d1", Content: "dog home"},
				{ID: "d2", Content: "dog food"},
				{ID: "d3", Content: "dog toy"},
				{ID: "d4", Content: "dog bed"},
			},
			query:      search.Query{Text: "zebra cat dog"},
			wantBM25:   []string{"rare", "both"},
			wantBucket: []string{"both", "rare"},
		},
		"typo: 打ち間違いで一致した珍しい語より、そのまま一致した語を上にする": {
			docs: []search.Doc{
				{ID: "typo", Content: "serach"},
				{ID: "exact", Content: "search"},
				{ID: "s1", Content: "search tips"},
				{ID: "s2", Content: "search tools"},
				{ID: "s3", Content: "search bar"},
			},
			query:      search.Query{Text: "search", Typo: true},
			wantBM25:   []string{"typo", "exact"},
			wantBucket: []string{"exact", "typo"},
		},
		"proximity: 短い文書より、語が隣り合う長い文書を上にする": {
			docs: []search.Doc{
				{ID: "far", Content: "quick brown fox"},
				{ID: "near", Content: "alpha beta gamma quick fox delta epsilon"},
			},
			query:      search.Query{Text: "quick fox"},
			wantBM25:   []string{"far", "near"},
			wantBucket: []string{"near", "far"},
		},
		"attribute: 本文に何度も出る文書より、タイトルに出る文書を上にする": {
			docs: []search.Doc{
				{ID: "content", Title: "notes", Content: "golang golang golang"},
				{ID: "title", Title: "golang", Content: "notes about tools"},
			},
			query:      search.Query{Text: "golang"},
			wantBM25:   []string{"content", "title"},
			wantBucket: []string{"title", "content"},
		},
		"exactness: 接頭辞で一致した語より、そのまま一致した語を上にする": {
			docs: []search.Doc{
				{ID: "prefix", Content: "seashell"},
				{ID: "exact", Content: "sea level rising fast"},
			},
			query:      search.Query{Text: "sea", Prefix: true},
			wantBM25:   []string{"prefix", "exact"},
			wantBucket: []string{"exact", "prefix"},
		},
		"規則の順: 語の数が多ければ打ち間違いがあっても上にする": {
			docs: []search.Doc{
				{ID: "one", Content: "search"},
				{ID: "two-typo", Content: "serach engine"},
			},
			query:      search.Query{Text: "search engine", Typo: true},
			wantBM25:   []string{"two-typo", "one"},
			wantBucket: []string{"two-typo", "one"},
		},
		"規則の順: 打ち間違いが少なければ語が離れていても上にする": {
			docs: []search.Doc{
				{ID: "typo-near", Content: "serach engine"},
				{ID: "exact-far", Content: "search alpha beta gamma engine"},
			},
			query:      search.Query{Text: "search engine", Typo: true},
			wantBM25:   []string{"typo-near", "exact-far"},
			wantBucket: []string{"exact-far", "typo-near"},
		},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			ix := search.New(tc.docs)
			bm25 := orderOf(ix.RankQuery(tc.query, -1).Hits, tc.wantBM25)
			q := tc.query
			q.Ranking = search.RankingBucket
			bucket := orderOf(ix.RankQuery(q, -1).Hits, tc.wantBM25)
			if !slices.Equal(bm25, tc.wantBM25) {
				t.Fatalf("BM25 = %v, want %v", bm25, tc.wantBM25)
			}
			if !slices.Equal(bucket, tc.wantBucket) {
				t.Fatalf("bucket = %v, want %v", bucket, tc.wantBucket)
			}
		})
	}
}

// orderOf は検索結果から ids の文書だけを結果の順に並べる。
func orderOf(hits []search.Hit, ids []string) []string {
	var out []string
	for _, h := range hits {
		if slices.Contains(ids, h.Doc.ID) {
			out = append(out, h.Doc.ID)
		}
	}
	return out
}

func TestRankBucketSharded(t *testing.T) {
	t.Parallel()
	ds := randomDocs(500)
	q := search.Query{Text: "w00003 w00010 w00200", Ranking: search.RankingBucket}
	want := search.New(ds).RankQuery(q, 10).Hits
	got := search.NewSharded(ds, 3, nil).RankQuery(q, 10).Hits
	for i := range want {
		if want[i].Score != got[i].Score {
			t.Fatalf("hit %d: sharded score %v, want %v", i, got[i].Score, want[i].Score)
		}
	}
}
