package search_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/hiro8ma/agent/go/internal/search"
)

// tiedScores は distinct 種類の点数だけを使い、同点を多く含む列を作る。withSpecial なら NaN と -Inf と +Inf を混ぜる。
func tiedScores(n, distinct int, withSpecial bool, seed uint64) []float64 {
	r := rand.New(rand.NewPCG(seed, uint64(n)))
	xs := make([]float64, n)
	for i := range xs {
		xs[i] = float64(r.IntN(distinct)) / 4
		if withSpecial && r.IntN(20) == 0 {
			xs[i] = []float64{math.NaN(), math.Inf(-1), math.Inf(1)}[r.IntN(3)]
		}
	}
	return xs
}

func TestTopKHeapMatchesSort(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		n, distinct int
		special     bool
	}{
		"候補 0 件":               {n: 0, distinct: 1},
		"候補 1 件":               {n: 1, distinct: 1},
		"すべて同点の 1,000 件":       {n: 1_000, distinct: 1},
		"3 種類の点数の 1,000 件":     {n: 1_000, distinct: 3},
		"20 種類の点数の 5,000 件":    {n: 5_000, distinct: 20},
		"ほぼ重ならない 5,000 件":      {n: 5_000, distinct: 1 << 20},
		"NaN と無限大を混ぜた 2,000 件": {n: 2_000, distinct: 5, special: true},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			for seed := range uint64(5) {
				scores := tiedScores(tc.n, tc.distinct, tc.special, seed)
				for _, k := range []int{0, 1, 2, 3, 10, 100, 999, tc.n, tc.n + 5} {
					want := search.TopKOrder(scores, k, false)
					got := search.TopKOrder(scores, k, true)
					if !slices.Equal(got, want) {
						t.Fatalf("seed %d k %d: heap = %v, sort = %v", seed, k, firstTen(got), firstTen(want))
					}
				}
			}
		})
	}
}

func firstTen(xs []int) []int { return xs[:min(len(xs), 10)] }

func BenchmarkTopK(b *testing.B) {
	for _, n := range []int{1_000, 10_000, 100_000} {
		scores := tiedScores(n, 1<<20, false, 1)
		for _, k := range []int{10, 100, 1_000} {
			if k >= n {
				continue
			}
			for _, heap := range []bool{false, true} {
				name := "sort"
				if heap {
					name = "heap"
				}
				b.Run(fmt.Sprintf("n=%d/k=%d/%s", n, k, name), func(b *testing.B) {
					for b.Loop() {
						search.TopKOrder(scores, k, heap)
					}
				})
			}
		}
	}
}

// BenchmarkTopKRatio は候補数と K の比を細かく変え、ヒープに切り替える比を決める。
func BenchmarkTopKRatio(b *testing.B) {
	for _, k := range []int{10, 100, 1_000} {
		for _, ratio := range []int{1, 2, 4, 8, 16} {
			n := k * ratio
			scores := tiedScores(n, 1<<20, false, 2)
			for _, heap := range []bool{false, true} {
				name := "sort"
				if heap {
					name = "heap"
				}
				b.Run(fmt.Sprintf("k=%d/ratio=%d/%s", k, ratio, name), func(b *testing.B) {
					for b.Loop() {
						search.TopKOrder(scores, k, heap)
					}
				})
			}
		}
	}
}
