package search_test

import (
	"testing"

	"github.com/hiro8ma/agent/go/internal/search"
)

func TestRankPhrase(t *testing.T) {
	t.Parallel()
	ds := []search.Doc{
		{Title: "ライス", Content: "ライスを炊く"},
		{Title: "カレー", Content: "カレーを煮込む"},
		{Title: "カレーライス", Content: "カレーライスを食べる"},
		{Title: "ライスカレー", Content: "ライスカレーを食べる"},
		{Title: "別々", Content: "カレーとライス"},
		{Title: "カレー", Content: "ライス"},
		{Title: "ばらばら", Content: "カレーとライスとルーラー"},
		{Title: "curry and rice", Content: "curry and rice"},
		{Title: "curry rice", Content: "curry rice"},
	}
	stop := search.WithAnalyzer(search.NewAnalyzer(search.WithStopWords(search.DefaultStopWords()...)))
	testCases := map[string]struct {
		opts       []search.Option
		query      search.Query
		wantScored int
		wantTop    string
	}{
		"フレーズでなければ bigram を 1 つでも含む文書が候補になる": {
			query: search.Query{Text: "カレーライス"}, wantScored: 7, wantTop: "カレーライス",
		},
		"フレーズなら bigram が連続して並ぶ文書だけが候補になる": {
			query: search.Query{Text: "カレーライス", Phrase: true}, wantScored: 1, wantTop: "カレーライス",
		},
		"フレーズはタイトルの末尾と本文の先頭をつないで照合しない": {
			query: search.Query{Text: "カレー ライス", Phrase: true}, wantScored: 0, wantTop: "",
		},
		"除去した語の位置は空けたまま照合する": {
			opts:  []search.Option{stop},
			query: search.Query{Text: "curry and rice", Phrase: true}, wantScored: 1, wantTop: "curry and rice",
		},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			res := search.New(ds, tc.opts...).RankQuery(tc.query, 1)
			if res.Scored != tc.wantScored {
				t.Fatalf("Scored = %d, want %d", res.Scored, tc.wantScored)
			}
			top := ""
			if len(res.Hits) > 0 {
				top = res.Hits[0].Doc.Title
			}
			if top != tc.wantTop {
				t.Fatalf("top = %q, want %q", top, tc.wantTop)
			}
		})
	}
}
