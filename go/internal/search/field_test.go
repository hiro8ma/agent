package search_test

import (
	"math"
	"slices"
	"testing"

	"github.com/hiro8ma/agent/go/internal/search"
)

func TestRankFieldWeights(t *testing.T) {
	t.Parallel()
	ds := []search.Doc{
		{Title: "夕食の献立", Content: "カレー カレー うどん"},
		{Title: "カレー", Content: "玉ねぎを炒めて肉と野菜を煮込み、ルーを溶かす"},
		{Title: "寿司", Content: "酢飯と魚"},
	}
	testCases := map[string]struct {
		opts       []search.Option
		fields     []search.Field
		wantTitles []string
	}{
		"本文を重くすると本文に多く含む文書が上": {
			opts:       []search.Option{search.WithFieldWeights(1, 3)},
			wantTitles: []string{"夕食の献立", "カレー"},
		},
		"タイトルを重くするとタイトルに含む文書が上": {
			opts:       []search.Option{search.WithFieldWeights(3, 1)},
			wantTitles: []string{"カレー", "夕食の献立"},
		},
		"タイトルだけを検索すると本文にだけ含む文書は候補にならない": {
			fields:     []search.Field{search.FieldTitle},
			wantTitles: []string{"カレー"},
		},
		"本文だけを検索するとタイトルにだけ含む文書は候補にならない": {
			fields:     []search.Field{search.FieldContent},
			wantTitles: []string{"夕食の献立"},
		},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			res := search.New(ds, tc.opts...).RankQuery(search.Query{Text: "カレー", Fields: tc.fields}, 10)
			if got := titles(res.Hits); !slices.Equal(got, tc.wantTitles) {
				t.Fatalf("titles = %v, want %v", got, tc.wantTitles)
			}
		})
	}
}

func TestRankWithoutFieldWeightsMatchesConcatenatedBM25(t *testing.T) {
	t.Parallel()
	ds := []search.Doc{{Title: "go", Content: "go x x x x x x x x x"}, {Title: "d1", Content: "go x"}}
	concat := make([]search.Doc, len(ds))
	for i, d := range ds {
		concat[i] = search.Doc{Content: d.Title + " " + d.Content}
	}
	got := search.New(ds).Rank("go", 10).Hits
	want := search.New(concat).Rank("go", 10).Hits
	if len(got) != len(want) {
		t.Fatalf("len = %d, want %d", len(got), len(want))
	}
	for i := range got {
		if math.Abs(got[i].Score-want[i].Score) > 1e-12 {
			t.Fatalf("score[%d] = %v, want %v", i, got[i].Score, want[i].Score)
		}
	}
}
