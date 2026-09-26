package search

import (
	"fmt"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"
)

func TestUnionIDs(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		a, b []int32
		want []int32
	}{
		"両方の要素を昇順に並べ、共通の要素は 1 つにする": {a: []int32{1, 3, 5, 12}, b: []int32{2, 3, 12, 13}, want: []int32{1, 2, 3, 5, 12, 13}},
		"片方が先に尽きたら残りをそのまま足す":        {a: []int32{1}, b: []int32{2, 4, 8}, want: []int32{1, 2, 4, 8}},
		"片方が空ならもう片方と同じ":             {a: []int32{}, b: []int32{7, 9}, want: []int32{7, 9}},
		"同じ列なら同じ列":                  {a: []int32{3, 7}, b: []int32{3, 7}, want: []int32{3, 7}},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			if got := unionIDs(tc.a, tc.b); !slices.Equal(got, tc.want) {
				t.Fatalf("got %v, want %v", got, tc.want)
			}
		})
	}
}

func TestIntersectIDs(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		ratio int
	}{
		"2 つのポインタ":  {ratio: 0},
		"galloping": {ratio: 1},
	}
	r := rand.New(rand.NewPCG(9, 9))
	pairs := [][2][]int32{
		{{3, 7, 8, 12, 13}, {1, 2, 3, 5, 12}},
		{sortedIDs(r, 10, 1_000), sortedIDs(r, 700, 1_000)},
		{sortedIDs(r, 300, 1_000), sortedIDs(r, 300, 1_000)},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			for _, p := range pairs {
				if got, want := intersectIDs(slices.Clone(p[0]), p[1], tc.ratio), naiveIntersect(p[0], p[1]); !slices.Equal(got, want) {
					t.Fatalf("got %v, want %v", got, want)
				}
			}
		})
	}
}

// matchMap は match を併合に変える前の実装で、マップで重複を除いてから並べ替える。比べるためだけに置く。
func (ix *Index) matchMap(terms []int32, sel fieldSet) []int {
	seen := make(map[int]struct{})
	var ids []int
	for _, t := range terms {
		for _, p := range ix.postingsOf(t) {
			if sel.tf(p) == 0 {
				continue
			}
			if _, ok := seen[int(p.doc)]; !ok {
				seen[int(p.doc)] = struct{}{}
				ids = append(ids, int(p.doc))
			}
		}
	}
	slices.Sort(ids)
	return ids
}

// zipfIndex は語の出現頻度が Zipf 分布に従う文書を n 件索引する。
func zipfIndex(n int) *Index {
	r := rand.New(rand.NewPCG(42, uint64(n)))
	zipf := rand.NewZipf(r, 1.1, 1, 19_999)
	ds := make([]Doc, n)
	for i := range ds {
		words := make([]string, 20+r.IntN(61))
		for j := range words {
			words[j] = fmt.Sprintf("w%05d", zipf.Uint64())
		}
		ds[i] = Doc{Content: strings.Join(words, " ")}
	}
	return New(ds)
}

var orQueries = map[string]string{
	"rare2":   "w00050 w02000",
	"common2": "w00001 w00002",
	"common5": "w00001 w00002 w00003 w00004 w00005",
	"mixed8":  "w00001 w00003 w00010 w00030 w00100 w00300 w01000 w03000",
}

func TestMatchEqualsMapUnion(t *testing.T) {
	t.Parallel()
	ix := zipfIndex(2_000)
	for qn, text := range orQueries {
		for _, fields := range [][]Field{nil, {FieldContent}, {FieldTitle}} {
			t.Run(fmt.Sprintf("%s/%v", qn, fields), func(t *testing.T) {
				t.Parallel()
				pq, sel := ix.parse(Query{Text: text}), selectFields(fields)
				if got, want := ix.match(pq.terms, sel), ix.matchMap(pq.terms, sel); !slices.Equal(got, want) {
					t.Fatalf("union %d ids, map %d ids", len(got), len(want))
				}
			})
		}
	}
}

// BenchmarkMatchOr は OR の候補を、マップで重複を除いて並べ替える matchMap と、並んだ列を前から併合する match で比べる。
//
//	go test -run '^$' -bench MatchOr -benchmem ./internal/search/
func BenchmarkMatchOr(b *testing.B) {
	for _, n := range []int{10_000, 100_000} {
		ix := zipfIndex(n)
		for qn, text := range orQueries {
			terms := ix.parse(Query{Text: text}).terms
			sel := selectFields(nil)
			for _, m := range []struct {
				name string
				f    func([]int32, fieldSet) []int
			}{{name: "map", f: ix.matchMap}, {name: "union", f: ix.match}} {
				b.Run(fmt.Sprintf("N=%d/%s/%s", n, qn, m.name), func(b *testing.B) {
					b.ReportAllocs()
					var got int
					for b.Loop() {
						got = len(m.f(terms, sel))
					}
					b.ReportMetric(float64(got), "candidates")
				})
			}
		}
	}
}
