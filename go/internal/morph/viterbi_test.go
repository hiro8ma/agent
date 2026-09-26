package morph_test

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/hiro8ma/agent/go/internal/morph"
)

var (
	textbookLabels = []string{"名詞", "動詞", "助詞"}
	textbookTokens = []string{"とうきょう", "と", "なら"}
)

// textbookProblem は教材の値を並べたもの。(と,名詞)=2 は教材に無く、「と」までの最小の名詞が 5 になることから決めた。
func textbookProblem() morph.Problem {
	return morph.Problem{
		Emit: [][]float64{
			{2, 20, 20},
			{2, 10, 1},
			{1, 2, 10},
		},
		Trans: [][]float64{
			{1, 1, 1},
			{2, 5, 1},
			{1, 1, 3},
		},
	}
}

func labelsOf(path []int, labels []string) []string {
	out := make([]string, len(path))
	for i, j := range path {
		out[i] = labels[j]
	}
	return out
}

func TestViterbiTextbook(t *testing.T) {
	t.Parallel()
	res := morph.Viterbi(textbookProblem())
	testCases := map[string]struct {
		pos  int
		want []float64
	}{
		"とうきょうまでの最小は割り当てのコストそのもの": {pos: 0, want: []float64{2, 20, 20}},
		"とまでの最小は名詞 5 動詞 13 助詞 4":  {pos: 1, want: []float64{5, 13, 4}},
		"ならまでの最小は名詞 6 動詞 7 助詞 16": {pos: 2, want: []float64{6, 7, 16}},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			if !slices.Equal(res.Min[tc.pos], tc.want) {
				t.Fatalf("%s: min = %v, want %v", textbookTokens[tc.pos], res.Min[tc.pos], tc.want)
			}
		})
	}
	if res.Cost != 6 {
		t.Fatalf("cost = %v, want 6", res.Cost)
	}
	if got, want := labelsOf(res.Path, textbookLabels), []string{"名詞", "助詞", "名詞"}; !slices.Equal(got, want) {
		t.Fatalf("path = %v, want %v", got, want)
	}
}

func TestViterbiTextbookVerbCandidates(t *testing.T) {
	t.Parallel()
	p := textbookProblem()
	testCases := map[string]struct {
		prev int
		want float64
	}{
		"とうきょうが名詞なら 2+1+10=13":  {prev: 0, want: 13},
		"とうきょうが動詞なら 20+5+10=35": {prev: 1, want: 35},
		"とうきょうが助詞なら 20+1+10=31": {prev: 2, want: 31},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			if got := morph.PathCost(p, []int{tc.prev, 1}); got != tc.want {
				t.Fatalf("cost = %v, want %v", got, tc.want)
			}
		})
	}
}

func randomProblem(r *rand.Rand, n, k int, withEnds bool) morph.Problem {
	mat := func(rows, cols int) [][]float64 {
		m := make([][]float64, rows)
		for i := range m {
			m[i] = make([]float64, cols)
			for j := range m[i] {
				m[i][j] = r.Float64()*10 - 3
			}
		}
		return m
	}
	p := morph.Problem{Emit: mat(n, k), Trans: mat(k, k)}
	if withEnds {
		ends := mat(2, k)
		p.Start, p.End = ends[0], ends[1]
	}
	return p
}

func TestViterbiMatchesBruteForce(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		n, k     int
		withEnds bool
	}{
		"1 語 3 品詞":         {n: 1, k: 3},
		"4 語 3 品詞":         {n: 4, k: 3},
		"6 語 4 品詞":         {n: 6, k: 4},
		"5 語 5 品詞で文頭と文末つき": {n: 5, k: 5, withEnds: true},
		"7 語 2 品詞で文頭と文末つき": {n: 7, k: 2, withEnds: true},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			r := rand.New(rand.NewPCG(uint64(tc.n), uint64(tc.k)))
			for range 50 {
				p := randomProblem(r, tc.n, tc.k, tc.withEnds)
				res := morph.Viterbi(p)
				_, want := morph.BruteForce(p)
				if math.Abs(res.Cost-want) > 1e-9 {
					t.Fatalf("viterbi = %v, brute force = %v", res.Cost, want)
				}
				if got := morph.PathCost(p, res.Path); math.Abs(got-res.Cost) > 1e-9 {
					t.Fatalf("path cost = %v, reported %v", got, res.Cost)
				}
			}
		})
	}
}
