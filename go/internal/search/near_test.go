package search_test

import (
	"slices"
	"testing"

	"github.com/hiro8ma/agent/go/internal/search"
)

// TestTutorialNear は教材の 3 文書で NEAR/k を確かめる。位置は 0 から数え、文書 1 のカレーは 0 と 5、ライスは 2、文書 3 のカレーは 2、ライスは 7。
func TestTutorialNear(t *testing.T) {
	t.Parallel()
	ix := whitespaceIndex(tutorialIndexDocs...)
	testCases := map[string]struct {
		text string
		near int
		want []string
	}{
		"辛いとカレーが 3 語以内なのは文書 2 だけ":          {text: "辛い カレー", near: 3, want: []string{"d1"}},
		"辛いとカレーは隣り合うので 1 語以内でも文書 2 が残る":    {text: "辛い カレー", near: 1, want: []string{"d1"}},
		"カレーとライスが 1 語以内の文書は無い":             {text: "カレー ライス", near: 1, want: []string{}},
		"文書 1 はカレーとライスの距離が 2 なので 2 語以内で残る": {text: "カレー ライス", near: 2, want: []string{"d0"}},
		"文書 3 は距離が 5 なので 4 語以内では外れる":       {text: "カレー ライス", near: 4, want: []string{"d0"}},
		"5 語以内なら文書 3 も残る":                  {text: "カレー ライス", near: 5, want: []string{"d0", "d2"}},
		"語の順番を入れ替えても同じ文書が残る":               {text: "ライス カレー", near: 2, want: []string{"d0"}},
		"3 語なら最初と最後の位置の差で測り文書 3 は差が 3":     {text: "カツ ライス 合う", near: 3, want: []string{"d2"}},
		"3 語で差が 2 以内の組は無い":                 {text: "カツ ライス 合う", near: 2, want: []string{}},
		"Near が 0 なら使わず OR の候補になる":         {text: "辛い カレー", near: 0, want: []string{"d0", "d1", "d2"}},
		"クエリの語が 1 つなら含む文書がすべて残る":           {text: "ライス", near: 1, want: []string{"d0", "d2"}},
		"どれかの語を含まない文書は位置を見る前に AND で落ちる":    {text: "スパイス ライス", near: 10, want: []string{}},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			got := hitDocIDs(ix.RankQuery(search.Query{Text: tc.text, Near: tc.near}, -1).Hits)
			slices.Sort(got)
			if !slices.Equal(got, tc.want) {
				t.Fatalf("ids = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestNearKeepsRemovedWordPositionsAndFields(t *testing.T) {
	t.Parallel()
	ds := []search.Doc{
		{ID: "and", Content: "curry and rice"},
		{ID: "adjacent", Content: "curry rice"},
		{ID: "split", Title: "tasty curry", Content: "rice dish"},
	}
	stop := search.WithAnalyzer(search.NewAnalyzer(search.WithStopWords(search.DefaultStopWords()...)))
	testCases := map[string]struct {
		near int
		want []string
	}{
		"除去した and の位置を空けたまま数えるので 1 語以内は隣り合う文書だけ": {near: 1, want: []string{"adjacent"}},
		"2 語以内なら and を挟む文書も残る":                   {near: 2, want: []string{"adjacent", "and"}},
		"タイトルと本文をつないで距離を測らない":                    {near: 10, want: []string{"adjacent", "and"}},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			got := hitDocIDs(search.New(ds, stop).RankQuery(search.Query{Text: "curry rice", Near: tc.near}, -1).Hits)
			slices.Sort(got)
			if !slices.Equal(got, tc.want) {
				t.Fatalf("ids = %v, want %v", got, tc.want)
			}
		})
	}
}
