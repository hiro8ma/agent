package search

import (
	"fmt"
	"math/rand/v2"
	"slices"
	"testing"
)

// intersectMap は短い ids をハッシュの集合にしてから長い ps を先頭から引く。ps が昇順なので結果も昇順になる。比べるためだけに置き、検索では使わない。
func intersectMap(ids []int32, ps []posting) []int32 {
	set := make(map[int32]struct{}, len(ids))
	for _, id := range ids {
		set[id] = struct{}{}
	}
	out := ids[:0]
	for _, p := range ps {
		if _, ok := set[p.doc]; ok {
			out = append(out, p.doc)
		}
	}
	return out
}

// gappedIDs は平均の間隔が universe/n になる昇順の文書番号を n 個作る。大きな列でも r.Perm の確保をしないで済む。
func gappedIDs(r *rand.Rand, n, universe int) []int32 {
	maxGap := max(2*universe/n-1, 1)
	ids := make([]int32, n)
	next := int32(r.IntN(maxGap))
	for i := range ids {
		ids[i] = next
		next += int32(1 + r.IntN(maxGap))
	}
	return ids
}

func TestIntersectMapMatchesNaive(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		short, long int
	}{
		"同じ長さ":       {short: 500, long: 500},
		"長さの比が 20":   {short: 100, long: 2_000},
		"長さの比が 1024": {short: 5, long: 5_120},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			r := rand.New(rand.NewPCG(7, uint64(tc.long)))
			a, b := sortedIDs(r, tc.short, 2*tc.long), sortedIDs(r, tc.long, 2*tc.long)
			want := naiveIntersect(a, b)
			if got := intersectMap(slices.Clone(a), postingsFor(b)); !slices.Equal(got, want) {
				t.Fatalf("got %d ids, want %d", len(got), len(want))
			}
		})
	}
}

// BenchmarkIntersectMethods は 2 つのポインタ / galloping / ハッシュの集合の 3 つで、短い方の長さと長さの比ごとに時間と確保量を比べる。
//
//	go test -run '^$' -bench IntersectMethods -benchmem ./internal/search/
func BenchmarkIntersectMethods(b *testing.B) {
	methods := []struct {
		name string
		f    func([]int32, []posting) []int32
	}{
		{name: "merge", f: func(ids []int32, ps []posting) []int32 { return intersectMerge(ids, ps, selectFields(nil), true) }},
		{name: "gallop", f: func(ids []int32, ps []posting) []int32 { return intersectGallop(ids, ps, selectFields(nil), true) }},
		{name: "map", f: intersectMap},
	}
	for _, short := range []int{100, 1_000, 10_000} {
		for _, ratio := range []int{1, 20, 1024} {
			long := short * ratio
			r := rand.New(rand.NewPCG(uint64(ratio), uint64(short)))
			a, ps := gappedIDs(r, short, 2*long), postingsFor(gappedIDs(r, long, 2*long))
			buf := make([]int32, short)
			for _, m := range methods {
				b.Run(fmt.Sprintf("short=%d/ratio=%d/%s", short, ratio, m.name), func(b *testing.B) {
					b.ReportAllocs()
					for b.Loop() {
						copy(buf, a)
						m.f(buf, ps)
					}
				})
			}
		}
	}
}
