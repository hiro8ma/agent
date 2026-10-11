package tei_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"testing"

	"github.com/hiro8ma/agent/go/internal/search"
)

// 保存した再ランクの結果だけを使い、TEI に繋がない。結果を作り直す手順は TestLiveRerankKenpou にある。

func mustRerankFixture(t *testing.T) (rerankFixture, map[string][][]int) {
	t.Helper()
	b, err := os.ReadFile(filepath.FromSlash(rerankFixturePath))
	if err != nil {
		t.Fatal(err)
	}
	var f rerankFixture
	if err := json.Unmarshal(b, &f); err != nil {
		t.Fatal(err)
	}
	if len(f.Queries) != len(searchQueries) {
		t.Fatalf("fixture has %d queries, want %d", len(f.Queries), len(searchQueries))
	}
	for i, q := range searchQueries {
		if f.Queries[i] != q.Text {
			t.Fatalf("fixture query %d is %q, want %q", i, f.Queries[i], q.Text)
		}
	}
	top := map[string][][]int{}
	for _, m := range f.Methods {
		if len(m.Top) != len(searchQueries) {
			t.Fatalf("%s has %d rankings, want %d", m.Key, len(m.Top), len(searchQueries))
		}
		top[m.Key] = m.Top
	}
	for _, m := range rerankMethods {
		if _, ok := top[m.Key]; !ok {
			t.Fatalf("fixture has no method %s", m.Key)
		}
	}
	return f, top
}

// TestRerankFromFixture は記録した上位の条から、クエリごとの最初の正解の順位を確かめる。0は上位10件に正解が無いこと。
// Ruri だけで hit@3 が1.00なので、hit@1 と MRR@10 の差も見る。
func TestRerankFromFixture(t *testing.T) {
	t.Parallel()
	f, top := mustRerankFixture(t)
	testCases := map[string]struct {
		key   string
		ranks []int
	}{
		"Ruri だけでは働く権利の正解が2位":                       {key: "ruri", ranks: []int{1, 1, 1, 1, 1, 1, 2, 1, 1, 1}},
		"上位20件の再ランクは働く権利を1位に上げ、好きな仕事を選べるを7位に下げる":    {key: "cross_top20", ranks: []int{1, 1, 1, 1, 1, 1, 1, 1, 7, 1}},
		"上位50件の再ランクは好きな仕事を選べるの正解を上位10件から落とす":        {key: "cross_top50", ranks: []int{1, 1, 1, 1, 1, 1, 1, 1, 0, 1}},
		"全トークンの MaxSim は Ruri と同じ順位":                {key: "maxsim", ranks: []int{1, 1, 1, 1, 1, 1, 2, 1, 1, 1}},
		"先頭と末尾のトークンを除いた MaxSim は差別されない権利の正解も2位に下げる": {key: "maxsim_inner", ranks: []int{2, 1, 1, 1, 1, 1, 2, 1, 1, 1}},
	}
	stored := map[string]float64{}
	for _, m := range f.Methods {
		stored[m.Key] = m.Hit3
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			got := make([]int, len(searchQueries))
			for qi, q := range searchQueries {
				for r, id := range top[tc.key][qi][:min(10, len(top[tc.key][qi]))] {
					if slices.Contains(q.Relevant, id) {
						got[qi] = r + 1
						break
					}
				}
			}
			if !slices.Equal(got, tc.ranks) {
				t.Errorf("%s ranks = %v, want %v", tc.key, got, tc.ranks)
			}
			if m := evaluate(top[tc.key]); stored[tc.key] != m.hit3 {
				t.Errorf("stored hit@3 = %.2f, recomputed %.2f", stored[tc.key], m.hit3)
			}
		})
	}
}

// TestRerankFixtureMatchesRuri は記録した Ruri の上位3件が保存した埋め込みと一致し、上位 n 件の再ランクが Ruri の上位 n 件の外を出していないことを確かめる。
func TestRerankFixtureMatchesRuri(t *testing.T) {
	t.Parallel()
	fx, articles := mustFixture(t)
	f, top := mustRerankFixture(t)
	docs := make([]search.Doc, len(articles))
	for i, a := range articles {
		docs[i] = search.Doc{ID: strconv.Itoa(i + 1), Content: a}
	}
	e := fixedEmbedder{}
	for _, q := range fx.Queries {
		e[q.Text] = q.Prefixed
	}
	flat, err := search.NewFlat(docs, fx.ArticlesPrefixed, e)
	if err != nil {
		t.Fatal(err)
	}
	for qi, q := range searchQueries {
		got, err := flat.Search(t.Context(), q.Text, slices.Max(f.Depths))
		if err != nil {
			t.Fatal(err)
		}
		ruri := docIDs(t, got)
		if want := top["ruri"][qi][:3]; !slices.Equal(ruri[:3], want) {
			t.Errorf("%s: 保存した埋め込みの上位3件 %v, 記録 %v", q.Text, ruri[:3], want)
		}
		for _, d := range f.Depths {
			for _, id := range top["cross_top"+strconv.Itoa(d)][qi] {
				if !slices.Contains(ruri[:d], id) {
					t.Errorf("%s: 上位%d件の再ランクの結果の第%d条が Ruri の上位%d件に無い", q.Text, d, id, d)
				}
			}
		}
	}
}
