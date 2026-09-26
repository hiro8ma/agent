package morph_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/hiro8ma/agent/go/internal/morph"
)

func BenchmarkViterbi(b *testing.B) {
	for _, k := range []int{3, 10, 30} {
		for _, n := range []int{5, 10, 15, 20} {
			p := randomProblem(rand.New(rand.NewPCG(1, 2)), n, k, false)
			b.Run(fmt.Sprintf("k=%d/n=%d/viterbi", k, n), func(b *testing.B) {
				for b.Loop() {
					morph.Viterbi(p)
				}
			})
			if math.Pow(float64(k), float64(n)) > 3e7 {
				continue
			}
			b.Run(fmt.Sprintf("k=%d/n=%d/brute", k, n), func(b *testing.B) {
				for b.Loop() {
					morph.BruteForce(p)
				}
			})
		}
	}
}
