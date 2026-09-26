package search

import (
	"slices"
	"testing"
)

// TestTutorialIntersectSteps は教材の手順（小さい方を進める、一致したら両方進める、どちらかの末尾で終わる）どおりに進むことを、止まった位置と比べた回数で確かめる。
// 1 回比べるごとにどちらかが 1 つ進み、一致したときだけ両方が進むので、比べた回数は止まった位置の和から一致の数を引いたものになる。
// 教材の組は (3,1) (3,2) (3,3) (7,5) (7,12) (8,12) (12,12) の 7 回で、カレーの末尾に達し、カツは 13 を読まずに止まる。
func TestTutorialIntersectSteps(t *testing.T) {
	t.Parallel()
	katsu := []int32{3, 7, 8, 12, 13}
	curry := []int32{1, 2, 3, 5, 12}
	testCases := map[string]struct {
		ids, ps      []int32
		want         []int32
		wantI, wantJ int
		wantN        int
	}{
		"カツとカレーはカレーの末尾で止まり 7 回":            {ids: katsu, ps: curry, want: []int32{3, 12}, wantI: 4, wantJ: 5, wantN: 7},
		"カレーとカツの順でもカレーの末尾で止まり 7 回":         {ids: curry, ps: katsu, want: []int32{3, 12}, wantI: 5, wantJ: 4, wantN: 7},
		"カツの 13 を落とすと 12 で両方の末尾に達して 7 回":   {ids: katsu[:4], ps: curry, want: []int32{3, 12}, wantI: 4, wantJ: 5, wantN: 7},
		"カレーの 12 を落とすと 5 の後にカツの 7 と比べて止まる": {ids: katsu, ps: curry[:4], want: []int32{3}, wantI: 1, wantJ: 4, wantN: 4},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			got, i, j := intersectMergeStop(slices.Clone(tc.ids), postingsFor(tc.ps), selectFields(nil), true)
			if !slices.Equal(got, tc.want) {
				t.Fatalf("got %v, want %v", got, tc.want)
			}
			if i != tc.wantI || j != tc.wantJ {
				t.Fatalf("stopped at (%d, %d), want (%d, %d)", i, j, tc.wantI, tc.wantJ)
			}
			if n := i + j - len(got); n != tc.wantN {
				t.Fatalf("comparisons = %d, want %d", n, tc.wantN)
			}
		})
	}
}

// TestTutorialPhraseSteps は教材の走査（5+1<10 で機械を進める、20+1>10 で学習を進める、20+1=21 で一致）の 3 回で止まることを確かめる。
func TestTutorialPhraseSteps(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		lists [][]int32
		offs  []int32
		want  bool
		wantN int
	}{
		"機械 学習は 20 と 21 で一致して 3 回で止まる": {
			lists: [][]int32{{5, 20, 50}, {10, 21, 43}}, offs: []int32{0, 1}, want: true, wantN: 3,
		},
		"学習の 21 が無ければ 50+1>43 で学習の末尾に達して一致しない": {
			lists: [][]int32{{5, 20, 50}, {10, 43}}, offs: []int32{0, 1}, want: false, wantN: 4,
		},
		"機械 学習 モデルは 2 語目で残った起点 20 だけを 3 語目の 6 と 22 に突き合わせる": {
			lists: [][]int32{{5, 20, 50}, {10, 21, 43}, {6, 22}}, offs: []int32{0, 1, 2}, want: true, wantN: 6,
		},
		"2 語目で残った起点 20 の 2 つ先に 3 語目が無ければ一致しない": {
			lists: [][]int32{{5, 20, 50}, {10, 21, 43}, {6, 23}}, offs: []int32{0, 1, 2}, want: false, wantN: 6,
		},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			got, n := phraseMatch(tc.lists, tc.offs)
			if got != tc.want {
				t.Fatalf("phraseMatch = %v, want %v", got, tc.want)
			}
			if n != tc.wantN {
				t.Fatalf("comparisons = %d, want %d", n, tc.wantN)
			}
		})
	}
}

// TestTutorialAndShortestFirst はクエリに書いた順ではなく、短い postings から順に共通部分を取ることを確かめる。
func TestTutorialAndShortestFirst(t *testing.T) {
	t.Parallel()
	ix := New([]Doc{
		{Content: "カレー おいしい"},
		{Content: "カレー カツ おいしい"},
		{Content: "カレー カツ"},
		{Content: "カレー おいしい"},
		{Content: "ライス"},
	}, WithAnalyzer(NewAnalyzer(WithWhitespaceTokenizer())))
	type step struct {
		n   int
		ids []int32
	}
	testCases := map[string]struct {
		text string
	}{
		"カツ AND カレー AND おいしい": {text: "カツ カレー おいしい"},
		"おいしい AND カレー AND カツ": {text: "おいしい カレー カツ"},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			pq := ix.parse(Query{Text: tc.text})
			lists := make([][]posting, len(pq.variants[0].slots))
			for i, slot := range pq.variants[0].slots {
				lists[i] = ix.slotPostings(pq, slot, selectFields(nil), true)
			}
			var steps []step
			got := intersectAll(lists, selectFields(nil), true, 0, func(n int, ids []int32) { steps = append(steps, step{n: n, ids: slices.Clone(ids)}) })
			want := []step{{n: 2, ids: []int32{1, 2}}, {n: 3, ids: []int32{1}}, {n: 4, ids: []int32{1}}}
			if !slices.EqualFunc(steps, want, func(a, b step) bool { return a.n == b.n && slices.Equal(a.ids, b.ids) }) {
				t.Fatalf("steps = %v, want %v", steps, want)
			}
			if !slices.Equal(got, []int32{1}) {
				t.Fatalf("got %v, want [1]", got)
			}
		})
	}
}
