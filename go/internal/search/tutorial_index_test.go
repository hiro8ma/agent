package search_test

import (
	"cmp"
	"fmt"
	"slices"
	"testing"

	"github.com/hiro8ma/agent/go/internal/search"
)

// whitespaceIndex は本文だけの文書を空白区切りで索引する。ID は d0 から振る。
func whitespaceIndex(contents ...string) *search.Index {
	ds := make([]search.Doc, len(contents))
	for i, c := range contents {
		ds[i] = search.Doc{ID: fmt.Sprintf("d%d", i), Content: c}
	}
	return search.New(ds, search.WithAnalyzer(search.NewAnalyzer(search.WithWhitespaceTokenizer())))
}

func hitDocIDs(hits []search.Hit) []string {
	ids := make([]string, len(hits))
	for i, h := range hits {
		ids[i] = h.Doc.ID
	}
	return ids
}

var tutorialIndexDocs = []string{
	"カレー と ライス は 合う カレー は おいしい",
	"スパイス の カレー は 辛い カレー は おいしい",
	"カツ と カレー は 合う カツ と ライス は 合う",
}

type bookPosting struct {
	doc int
	tf  int
	pos []int32
}

// TestTutorialInvertedIndex は教材の表と比べる。教材は文書と位置を 1 から数え、索引は 0 から数えるので、どちらも 1 を足して比べる。
func TestTutorialInvertedIndex(t *testing.T) {
	t.Parallel()
	ix := whitespaceIndex(tutorialIndexDocs...)
	testCases := map[string]struct {
		term string
		want []bookPosting
	}{
		"カレーは文書 1 と 2 で 2 回、文書 3 で 1 回（文書と位置は 1 から数える）": {term: "カレー", want: []bookPosting{{doc: 1, tf: 2, pos: []int32{1, 6}}, {doc: 2, tf: 2, pos: []int32{3, 6}}, {doc: 3, tf: 1, pos: []int32{3}}}},
		"ライスは文書 1 と 3 に 1 回ずつ（文書と位置は 1 から数える）":          {term: "ライス", want: []bookPosting{{doc: 1, tf: 1, pos: []int32{3}}, {doc: 3, tf: 1, pos: []int32{8}}}},
		"はは各文書に 2 回（文書と位置は 1 から数える）":                    {term: "は", want: []bookPosting{{doc: 1, tf: 2, pos: []int32{4, 7}}, {doc: 2, tf: 2, pos: []int32{4, 7}}, {doc: 3, tf: 2, pos: []int32{4, 9}}}},
		"おいしいは文書 1 と 2 の 8 語目（文書と位置は 1 から数える）":          {term: "おいしい", want: []bookPosting{{doc: 1, tf: 1, pos: []int32{8}}, {doc: 2, tf: 1, pos: []int32{8}}}},
		"とは文書 1 に 1 回、文書 3 に 2 回（文書と位置は 1 から数える）":       {term: "と", want: []bookPosting{{doc: 1, tf: 1, pos: []int32{2}}, {doc: 3, tf: 2, pos: []int32{2, 7}}}},
		"合うは文書 1 に 1 回、文書 3 に 2 回（文書と位置は 1 から数える）":      {term: "合う", want: []bookPosting{{doc: 1, tf: 1, pos: []int32{5}}, {doc: 3, tf: 2, pos: []int32{5, 10}}}},
		"カツは文書 3 だけに 2 回（文書と位置は 1 から数える）":               {term: "カツ", want: []bookPosting{{doc: 3, tf: 2, pos: []int32{1, 6}}}},
		"スパイスは文書 2 の先頭（文書と位置は 1 から数える）":                 {term: "スパイス", want: []bookPosting{{doc: 2, tf: 1, pos: []int32{1}}}},
		"のは文書 2 の 2 語目（文書と位置は 1 から数える）":                 {term: "の", want: []bookPosting{{doc: 2, tf: 1, pos: []int32{2}}}},
		"辛いは文書 2 の 5 語目（文書と位置は 1 から数える）":                {term: "辛い", want: []bookPosting{{doc: 2, tf: 1, pos: []int32{5}}}},
		"文書に無い語の postings は空":                           {term: "ナン", want: []bookPosting{}},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			ps := search.PostingsOf(ix, tc.term)
			got := make([]bookPosting, len(ps))
			for i, p := range ps {
				pos := make([]int32, len(p.Positions))
				for j, x := range p.Positions {
					pos[j] = x + 1
				}
				got[i] = bookPosting{doc: p.Doc + 1, tf: p.TF, pos: pos}
			}
			if !slices.EqualFunc(got, tc.want, func(a, b bookPosting) bool {
				return a.doc == b.doc && a.tf == b.tf && slices.Equal(a.pos, b.pos)
			}) {
				t.Fatalf("postings = %v, want %v", got, tc.want)
			}
		})
	}
	if got := ix.Stats().Terms; got != 10 {
		t.Fatalf("Terms = %d, want 10", got)
	}
}

func TestTutorialBooleanQuery(t *testing.T) {
	t.Parallel()
	ix := whitespaceIndex(tutorialIndexDocs...)
	testCases := map[string]struct {
		query search.Query
		want  []string
	}{
		"カレー AND ライスは文書 1 と 3（文書は 1 から数える）":      {query: search.Query{Text: "カレー ライス", Operator: search.OperatorAnd}, want: []string{"d0", "d2"}},
		"カレー OR ライスは 3 件すべて":                     {query: search.Query{Text: "カレー ライス"}, want: []string{"d0", "d1", "d2"}},
		"スパイス AND ライスはどの文書にも無い":                  {query: search.Query{Text: "スパイス ライス", Operator: search.OperatorAnd}, want: []string{}},
		"カツ AND 合う AND ライスは文書 3 だけ（文書は 1 から数える）": {query: search.Query{Text: "カツ 合う ライス", Operator: search.OperatorAnd}, want: []string{"d2"}},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			got := hitDocIDs(ix.RankQuery(tc.query, -1).Hits)
			slices.Sort(got)
			if !slices.Equal(got, tc.want) {
				t.Fatalf("titles = %v, want %v", got, tc.want)
			}
		})
	}
}

type countHit struct {
	doc   int
	count int
}

// matchCount はクエリの語のうち文書に含まれる語の数を点数にし、1 語も含まない文書は落とす。
func matchCount(ix *search.Index, terms ...string) []countHit {
	counts := map[int]int{}
	for _, term := range terms {
		for _, p := range search.PostingsOf(ix, term) {
			counts[p.Doc]++
		}
	}
	hits := make([]countHit, 0, len(counts))
	for doc, n := range counts {
		hits = append(hits, countHit{doc: doc, count: n})
	}
	slices.SortFunc(hits, func(a, b countHit) int { return cmp.Or(cmp.Compare(b.count, a.count), cmp.Compare(a.doc, b.doc)) })
	return hits
}

func TestTutorialMatchCountScore(t *testing.T) {
	t.Parallel()
	ix := whitespaceIndex(
		"転置インデックス は 素晴らしい",
		"転置インデックス は 速い",
		"検索 は 素晴らしい",
		"カレー は おいしい",
	)
	testCases := map[string]struct {
		terms []string
		want  []countHit
	}{
		"2 語とも含む文書が 2 点で 1 位、片方だけの文書は 1 点": {
			terms: []string{"素晴らしい", "転置インデックス"},
			want:  []countHit{{doc: 0, count: 2}, {doc: 1, count: 1}, {doc: 2, count: 1}},
		},
		"素晴らしいだけなら含む 2 件が 1 点ずつ": {
			terms: []string{"素晴らしい"},
			want:  []countHit{{doc: 0, count: 1}, {doc: 2, count: 1}},
		},
		"どの文書にも無い語なら 0 件": {
			terms: []string{"ねこ"},
			want:  []countHit{},
		},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			if got := matchCount(ix, tc.terms...); !slices.Equal(got, tc.want) {
				t.Fatalf("hits = %v, want %v", got, tc.want)
			}
		})
	}
}
