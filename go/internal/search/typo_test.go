package search_test

import (
	"slices"
	"testing"

	"github.com/hiro8ma/agent/go/internal/search"
)

func TestRankTypo(t *testing.T) {
	t.Parallel()
	ix := search.New([]search.Doc{
		{ID: "search", Content: "search engine basics"},
		{ID: "index", Content: "inverted index structure"},
		{ID: "retrieval", Content: "information retrieval systems"},
		{ID: "shells", Content: "sea shells"},
		{ID: "ja", Content: "検索エンジンの仕組み"},
	})
	testCases := map[string]struct {
		query search.Query
		want  []string
	}{
		"許容なしでは serach は 0 件":                {query: search.Query{Text: "serach"}, want: []string{}},
		"許容ありなら serach で search を見つける":       {query: search.Query{Text: "serach", Typo: true}, want: []string{"search"}},
		"許容なしでは indxe は 0 件":                 {query: search.Query{Text: "indxe"}, want: []string{}},
		"許容ありなら indxe で index を見つける":         {query: search.Query{Text: "indxe", Typo: true}, want: []string{"index"}},
		"許容なしでは retreival は 0 件":             {query: search.Query{Text: "retreival"}, want: []string{}},
		"許容ありなら retreival で retrieval を見つける": {query: search.Query{Text: "retreival", Typo: true}, want: []string{"retrieval"}},
		"6 文字の語の最初の文字の打ち間違いは許さない":            {query: search.Query{Text: "xearch", Typo: true}, want: []string{}},
		"9 文字の語なら最初の文字の打ち間違いも許す":             {query: search.Query{Text: "xetrieval", Typo: true}, want: []string{"retrieval"}},
		"4 文字の語は打ち間違いを許さない":                  {query: search.Query{Text: "shelk", Typo: true}, want: []string{}},
		"接頭辞なしでは入力途中の shel は 0 件":            {query: search.Query{Text: "shel"}, want: []string{}},
		"接頭辞ありなら最後の語 shel で shells を見つける":    {query: search.Query{Text: "shel", Prefix: true}, want: []string{"shells"}},
		"接頭辞は最後の語だけに使う":                      {query: search.Query{Text: "shel engine", Prefix: true}, want: []string{"search"}},
		"接頭辞と許容を合わせると入力途中の打ち間違いも拾う":          {query: search.Query{Text: "serac", Typo: true, Prefix: true}, want: []string{"search"}},
		"AND でも打ち間違いの語を 1 語として数える":           {query: search.Query{Text: "serach engine", Typo: true, Operator: search.OperatorAnd}, want: []string{"search"}},
		"フレーズでも打ち間違いの語の位置で照合する":              {query: search.Query{Text: "inverted indxe", Typo: true, Phrase: true}, want: []string{"index"}},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			got := hitDocIDs(ix.RankQuery(tc.query, -1).Hits)
			slices.Sort(got)
			if !slices.Equal(got, tc.want) {
				t.Fatalf("ids = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestRankTypoLeavesJapaneseAlone(t *testing.T) {
	t.Parallel()
	ix := search.New([]search.Doc{{ID: "a", Content: "検索エンジンの仕組み"}, {ID: "b", Content: "エンジニアの仕事"}})
	for _, text := range []string{"検索エンジソ", "エンジ", "仕"} {
		plain := ix.RankQuery(search.Query{Text: text}, -1)
		typo := ix.RankQuery(search.Query{Text: text, Typo: true, Prefix: true}, -1)
		if !slices.Equal(hitDocIDs(plain.Hits), hitDocIDs(typo.Hits)) || plain.Scored != typo.Scored {
			t.Fatalf("%q: typo %v, plain %v", text, hitDocIDs(typo.Hits), hitDocIDs(plain.Hits))
		}
	}
}
