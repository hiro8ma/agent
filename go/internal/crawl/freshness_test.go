package crawl_test

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/hiro8ma/agent/go/internal/crawl"
)

// logNormalRates は平均 mean、対数の標準偏差 sigma の対数正規分布から変化の頻度を n 件引く。
func logNormalRates(n int, mean, sigma float64, seed uint64) []float64 {
	r := rand.New(rand.NewPCG(seed, 2))
	mu := math.Log(mean) - sigma*sigma/2
	out := make([]float64, n)
	for i := range out {
		out[i] = math.Exp(mu + sigma*r.NormFloat64())
	}
	return out
}

func sum(xs []float64) float64 {
	var s float64
	for _, x := range xs {
		s += x
	}
	return s
}

func TestFreshnessFormula(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		lambda, f float64
		want      float64
	}{
		"取り直さなければ0":        {lambda: 1, f: 0, want: 0},
		"変わらなければ1":         {lambda: 0, f: 1, want: 1},
		"変化と取り直しが同じ頻度":     {lambda: 2, f: 2, want: 1 - math.Exp(-1)},
		"変化が取り直しの9倍":       {lambda: 9, f: 1, want: (1 - math.Exp(-9)) / 9},
		"取り直しが変化の10倍ならほぼ1": {lambda: 0.1, f: 1, want: (1 - math.Exp(-0.1)) / 0.1},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			if got := crawl.Freshness(tc.lambda, tc.f); math.Abs(got-tc.want) > 1e-12 {
				t.Errorf("Freshness(%v, %v) = %v, want %v", tc.lambda, tc.f, got, tc.want)
			}
		})
	}
}

// 許す差0.01は、400回の取り直しを500ページで平均した模擬の標準誤差（0.002以下）の5倍に取った。
func TestSimulatedFreshnessMatchesFormula(t *testing.T) {
	t.Parallel()
	lambdas := logNormalRates(500, 1, 1.5, 7)
	budget := float64(len(lambdas))
	testCases := map[string]struct {
		allocate func([]float64, float64) []float64
	}{
		"一様": {allocate: crawl.UniformAllocation},
		"比例": {allocate: crawl.ProportionalAllocation},
		"最適": {allocate: crawl.OptimalAllocation},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			freqs := tc.allocate(lambdas, budget)
			want := crawl.MeanFreshness(lambdas, freqs)
			got := crawl.SimulateFreshness(lambdas, freqs, 400, 11)
			if math.Abs(got-want) > 0.01 {
				t.Errorf("模擬 %.4f, 式 %.4f", got, want)
			}
		})
	}
}

func TestSimulatedFreshnessSinglePage(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		lambda, f float64
	}{
		"変化が取り直しの10分の1": {lambda: 0.1, f: 1},
		"変化と取り直しが同じ頻度":  {lambda: 1, f: 1},
		"変化が取り直しの5倍":    {lambda: 5, f: 1},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			want := crawl.Freshness(tc.lambda, tc.f)
			got := crawl.SimulateFreshness([]float64{tc.lambda}, []float64{tc.f}, 40_000, 3)
			if math.Abs(got-want) > 0.01 {
				t.Errorf("模擬 %.4f, 式 %.4f", got, want)
			}
		})
	}
}

func TestOptimalAllocationKeepsBudget(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		lambdas []float64
		budget  float64
	}{
		"対数正規の1,000ページに1ページあたり0.25回": {lambdas: logNormalRates(1_000, 1, 1.5, 1), budget: 250},
		"対数正規の1,000ページに1ページあたり4回":    {lambdas: logNormalRates(1_000, 1, 1.5, 1), budget: 4_000},
		"同じ頻度の3ページ":                  {lambdas: []float64{2, 2, 2}, budget: 3},
		"頻度が大きく違う2ページ":               {lambdas: []float64{0.01, 50}, budget: 0.5},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			freqs := crawl.OptimalAllocation(tc.lambdas, tc.budget)
			if got := sum(freqs); math.Abs(got-tc.budget) > 1e-9*tc.budget {
				t.Errorf("回数の和 = %.12g, want %.12g", got, tc.budget)
			}
			for i, f := range freqs {
				if f < 0 {
					t.Errorf("ページ %d に負の回数 %v", i, f)
				}
			}
		})
	}
}

func TestOptimalAllocationSkipsFastPages(t *testing.T) {
	t.Parallel()
	freqs := crawl.OptimalAllocation([]float64{1, 100}, 1)
	if freqs[1] != 0 {
		t.Errorf("1日に100回変わるページに %v 回を配った。予算1回では0回が最適", freqs[1])
	}
	if math.Abs(freqs[0]-1) > 1e-9 {
		t.Errorf("1日に1回変わるページ = %v, want 1", freqs[0])
	}
}

func TestUniformBeatsProportional(t *testing.T) {
	t.Parallel()
	lambdas := []float64{1, 9}
	uniform := crawl.MeanFreshness(lambdas, crawl.UniformAllocation(lambdas, 2))
	proportional := crawl.MeanFreshness(lambdas, crawl.ProportionalAllocation(lambdas, 2))
	if want := ((1 - math.Exp(-1)) + (1-math.Exp(-9))/9) / 2; math.Abs(uniform-want) > 1e-12 {
		t.Errorf("一様 = %v, want %v", uniform, want)
	}
	if want := (1 - math.Exp(-5)) / 5; math.Abs(proportional-want) > 1e-12 {
		t.Errorf("比例 = %v, want %v", proportional, want)
	}
	if uniform <= proportional {
		t.Errorf("一様 %.4f が比例 %.4f 以下", uniform, proportional)
	}
}

func TestOptimalBeatsOtherAllocations(t *testing.T) {
	t.Parallel()
	lambdas := []float64{0.2, 0.7, 1.5, 4, 12}
	const budget = 5.0
	best := crawl.MeanFreshness(lambdas, crawl.OptimalAllocation(lambdas, budget))
	r := rand.New(rand.NewPCG(5, 5))
	candidates := [][]float64{
		crawl.UniformAllocation(lambdas, budget),
		crawl.ProportionalAllocation(lambdas, budget),
	}
	for range 500 {
		w := make([]float64, len(lambdas))
		for i := range w {
			w[i] = r.ExpFloat64()
		}
		total := sum(w)
		for i := range w {
			w[i] *= budget / total
		}
		candidates = append(candidates, w)
	}
	for _, c := range candidates {
		if got := crawl.MeanFreshness(lambdas, c); got > best+1e-12 {
			t.Fatalf("配分 %v の鮮度 %.6f が最適 %.6f を上回る", c, got, best)
		}
	}
}

// marginalGain は Freshness の数値微分で、取り直しを1回増やしたときの鮮度の伸びを返す。
func marginalGain(lambda, f float64) float64 {
	h := 1e-6 * f
	return (crawl.Freshness(lambda, f+h) - crawl.Freshness(lambda, f-h)) / (2 * h)
}

func TestOptimalAllocationEqualizesMarginalGain(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		lambdas []float64
		budget  float64
	}{
		"対数正規の200ページに1ページあたり0.25回": {lambdas: logNormalRates(200, 1, 1.5, 3), budget: 50},
		"対数正規の200ページに1ページあたり1回":    {lambdas: logNormalRates(200, 1, 1.5, 3), budget: 200},
		"対数正規の200ページに1ページあたり4回":    {lambdas: logNormalRates(200, 1, 1.5, 3), budget: 800},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			freqs := crawl.OptimalAllocation(tc.lambdas, tc.budget)
			lo, hi := math.Inf(1), 0.0
			for i, f := range freqs {
				if f > 0 {
					g := marginalGain(tc.lambdas[i], f)
					lo, hi = min(lo, g), max(hi, g)
				}
			}
			if hi/lo-1 > 1e-4 {
				t.Errorf("回数が正のページの伸びが %.6g〜%.6g でそろわない", lo, hi)
			}
			for i, f := range freqs {
				// 0回のページの、0からの1回目の伸びは 1/λ。これが正のページの伸びを上回るなら、そこへ回すほうが鮮度が上がる。
				if f == 0 && 1/tc.lambdas[i] > hi*(1+1e-4) {
					t.Errorf("0回のページ %d（λ=%.3f）の伸び %.6g が、正のページの伸び %.6g を上回る", i, tc.lambdas[i], 1/tc.lambdas[i], hi)
				}
			}
		})
	}
}
