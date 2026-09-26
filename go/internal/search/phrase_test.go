package search_test

import (
	"slices"
	"strings"
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

// TestRankPhraseMatchesNaive はフレーズの候補が、文書の語の並びを 1 件ずつ調べた結果と一致することを確かめる。
func TestRankPhraseMatchesNaive(t *testing.T) {
	t.Parallel()
	ds := randomDocs(3_000)
	ix := search.New(ds)
	testCases := map[string]struct {
		text string
	}{
		"頻出の 2 語":      {text: "w00001 w00002"},
		"頻出の 2 語を逆の順に": {text: "w00002 w00001"},
		"同じ語が続く":       {text: "w00001 w00001"},
		"3 語":          {text: "w00001 w00002 w00001"},
		"3 語で間に別の語":    {text: "w00003 w00001 w00002"},
		"頻度の低い語を含む":    {text: "w00001 w00050"},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			q := strings.Fields(tc.text)
			want := []string{}
			for _, d := range ds {
				words := strings.Fields(d.Content)
				for i := range len(words) - len(q) + 1 {
					if slices.Equal(words[i:i+len(q)], q) {
						want = append(want, d.Title)
						break
					}
				}
			}
			got := titles(ix.RankQuery(search.Query{Text: tc.text, Phrase: true}, -1).Hits)
			slices.Sort(got)
			slices.Sort(want)
			if !slices.Equal(got, want) {
				t.Fatalf("got %d docs, want %d", len(got), len(want))
			}
			if len(want) == 0 {
				t.Fatal("no doc has the phrase; pick more frequent words")
			}
		})
	}
}
