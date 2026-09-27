package search_test

import (
	"fmt"
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/hiro8ma/agent/go/internal/search"
)

var saturationTF = []int{1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 1000}

// tfPart は b=0 の索引で語 t を f 回だけ含む文書の点を、f=1 の文書の点で割る。f=1 の TF の部分は (k1+1)/(1+k1) = 1 なので、比がそのまま f の TF の部分になる。
func tfPart(t *testing.T, k1 float64) map[int]float64 {
	t.Helper()
	ds := make([]search.Doc, len(saturationTF))
	for i, f := range saturationTF {
		ds[i] = search.Doc{ID: fmt.Sprint(f), Content: strings.TrimSpace(strings.Repeat("t ", f))}
	}
	ix := wsIndex(ds, search.WithK1(k1), search.WithB(0))
	score := make(map[int]float64)
	for _, h := range ix.Rank("t", -1).Hits {
		var f int
		if _, err := fmt.Sscan(h.Doc.ID, &f); err != nil {
			t.Fatal(err)
		}
		score[f] = h.Score
	}
	part := make(map[int]float64, len(score))
	for f, s := range score {
		part[f] = s / score[1]
	}
	return part
}

func TestBM25TFSaturation(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		k1 float64
	}{
		"k1=0.5": {k1: 0.5},
		"k1=1.2": {k1: 1.2},
		"k1=2.0": {k1: 2.0},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			part := tfPart(t, tc.k1)
			limit := tc.k1 + 1
			for i, f := range saturationTF {
				want := float64(f) * (tc.k1 + 1) / (float64(f) + tc.k1)
				if math.Abs(part[f]-want) > 1e-9 {
					t.Fatalf("f=%d: tf part = %v, want %v", f, part[f], want)
				}
				if part[f] >= limit {
					t.Fatalf("f=%d: tf part %v reaches k1+1 = %v", f, part[f], limit)
				}
				if i > 0 && part[f] <= part[saturationTF[i-1]] {
					t.Fatalf("f=%d: tf part %v does not grow from %v", f, part[f], part[saturationTF[i-1]])
				}
			}
			if gap := limit - part[1000]; gap > 0.01 {
				t.Fatalf("f=1000: tf part %v is %v below k1+1", part[1000], gap)
			}
			if d1, d2 := part[2]-part[1], part[3]-part[2]; d1 <= d2 {
				t.Fatalf("increment 1->2 = %v, 2->3 = %v, want diminishing", d1, d2)
			}
		})
	}
}

// lengthDocs は語 t を 1 回ずつ含み、pad の数で長さだけを変えた文書を作る。
func lengthDocs() []search.Doc {
	lengths := []int{1, 5, 10, 40, 100}
	ds := make([]search.Doc, len(lengths))
	for i, n := range lengths {
		ds[i] = search.Doc{ID: fmt.Sprint(n), Content: "t" + strings.Repeat(" pad", n-1)}
	}
	return ds
}

func TestBM25DocumentLength(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		b         float64
		wantEqual bool
	}{
		"b=0 は文書長を見ず、長さを変えても点が同じ":       {b: 0, wantEqual: true},
		"b=0.75 は長い文書ほど点が下がる":           {b: 0.75},
		"b=1 は長い文書ほど点が下がり、平均より長い文書は下がる": {b: 1},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			ds := lengthDocs()
			hits := wsIndex(ds, search.WithB(tc.b)).Rank("t", -1).Hits
			score := make(map[string]float64)
			for _, h := range hits {
				score[h.Doc.ID] = h.Score
			}
			for i := 1; i < len(ds); i++ {
				prev, cur := score[ds[i-1].ID], score[ds[i].ID]
				if tc.wantEqual && math.Abs(prev-cur) > 1e-12 {
					t.Fatalf("len %s = %v, len %s = %v, want equal", ds[i-1].ID, prev, ds[i].ID, cur)
				}
				if !tc.wantEqual && cur >= prev {
					t.Fatalf("len %s = %v, len %s = %v, want lower for the longer", ds[i-1].ID, prev, ds[i].ID, cur)
				}
			}
		})
	}
	t.Run("b=1 では平均より長い文書の点が b=0 より低く、平均より短い文書は高い", func(t *testing.T) {
		t.Parallel()
		ds := lengthDocs()
		at := func(b float64) map[string]float64 {
			out := make(map[string]float64)
			for _, h := range wsIndex(ds, search.WithB(b)).Rank("t", -1).Hits {
				out[h.Doc.ID] = h.Score
			}
			return out
		}
		b0, b1 := at(0), at(1)
		if b1["100"] >= b0["100"] || b1["1"] <= b0["1"] {
			t.Fatalf("len 100: b=1 %v vs b=0 %v, len 1: b=1 %v vs b=0 %v", b1["100"], b0["100"], b1["1"], b0["1"])
		}
	})
}

func TestBM25IDFLogBase(t *testing.T) {
	t.Parallel()
	ds := randomDocs(2_000)
	natural, base2 := search.New(ds), search.New(ds, search.WithLog2IDF())
	testCases := map[string]struct {
		query search.Query
	}{
		"2 語":       {query: search.Query{Text: benchQuery}},
		"3 語":       {query: search.Query{Text: "w00001 w00300 w01234"}},
		"AND":       {query: search.Query{Text: "w00001 w00002", Operator: search.OperatorAnd}},
		"タイトルと本文の語": {query: search.Query{Text: "doc1 w00050"}},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			a, b := natural.RankQuery(tc.query, -1).Hits, base2.RankQuery(tc.query, -1).Hits
			if len(a) == 0 || !slices.Equal(hitTitles(a), hitTitles(b)) {
				t.Fatalf("order differs: %d hits vs %d hits", len(a), len(b))
			}
			for i := range a {
				if math.Abs(b[i].Score/a[i].Score-1/math.Ln2) > 1e-9 {
					t.Fatalf("hit %d: log2 / ln = %v, want 1/ln2 = %v", i, b[i].Score/a[i].Score, 1/math.Ln2)
				}
			}
		})
	}
}
