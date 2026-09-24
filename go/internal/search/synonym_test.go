package search_test

import (
	"slices"
	"testing"

	"github.com/hiro8ma/agent/go/internal/search"
)

func TestSynonymsAtIndexTimeAndQueryExpansion(t *testing.T) {
	t.Parallel()
	ds := []search.Doc{
		{Title: "サインインの手順", Content: "メールアドレスでサインインする"},
		{Title: "ログインできない", Content: "ログインに失敗したらパスワードを再設定する"},
		{Title: "請求書", Content: "請求書をダウンロードする"},
	}
	dict := map[string]string{"サインイン": "ログイン"}
	plain := search.New(ds)
	indexTime := search.New(ds, search.WithAnalyzer(search.NewAnalyzer(search.WithSynonyms(dict))))
	testCases := map[string]struct {
		ix         *search.Index
		query      search.Query
		wantTitles []string // 順位ではなく一致した文書の集合を比べるため、名前の順に並べる
	}{
		"辞書が無ければログインのフレーズでサインインの文書は見つからない": {
			ix:         plain,
			query:      search.Query{Text: "ログイン", Phrase: true},
			wantTitles: []string{"ログインできない"},
		},
		"索引時にそろえるとログインでサインインの文書が見つかる": {
			ix:         indexTime,
			query:      search.Query{Text: "ログイン", Phrase: true},
			wantTitles: []string{"サインインの手順", "ログインできない"},
		},
		"クエリを広げると索引を作り直さずにサインインの文書が見つかる": {
			ix:         plain,
			query:      search.Query{Text: "ログイン", Phrase: true, Synonyms: dict},
			wantTitles: []string{"サインインの手順", "ログインできない"},
		},
		"クエリを広げる辞書は置き換え先からも見出しを引く": {
			ix:         plain,
			query:      search.Query{Text: "サインイン", Phrase: true, Synonyms: dict},
			wantTitles: []string{"サインインの手順", "ログインできない"},
		},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			if got := slices.Sorted(slices.Values(titles(tc.ix.RankQuery(tc.query, 10).Hits))); !slices.Equal(got, tc.wantTitles) {
				t.Fatalf("titles = %v, want %v", got, tc.wantTitles)
			}
		})
	}
}

func TestQueryExpansionTakesBestVariant(t *testing.T) {
	t.Parallel()
	ix := search.New([]search.Doc{
		{Title: "ログイン", Content: "ログインに失敗する"},
		{Title: "サインイン", Content: "サインインの手順"},
		{Title: "請求書", Content: "請求書をダウンロードする"},
	})
	plain := ix.RankQuery(search.Query{Text: "ログイン"}, 1).Hits[0]
	expanded := ix.RankQuery(search.Query{Text: "ログイン", Synonyms: map[string]string{"サインイン": "ログイン"}}, 3)
	for _, h := range expanded.Hits {
		if h.Doc.Title == plain.Doc.Title && h.Score != plain.Score {
			t.Fatalf("expanded score = %v, want %v (the score of the matching variant)", h.Score, plain.Score)
		}
	}
}
