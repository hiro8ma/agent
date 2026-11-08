package crawl

import (
	"math"
	"math/rand/v2"
)

// Freshness は、1日に lambda 回のポアソン過程で変わるページを、1日に f 回の一定の間隔 I = 1/f で取り直したときに、取ってある内容が最新である時間の割合の期待値 (1 − e^(−λI)) / (λI) を返す。
// f が0なら0、lambda が0なら1。
func Freshness(lambda, f float64) float64 {
	if f <= 0 {
		return 0
	}
	if lambda <= 0 {
		return 1
	}
	x := lambda / f
	return -math.Expm1(-x) / x
}

// MeanFreshness はページごとの Freshness の平均を返す。
func MeanFreshness(lambdas, freqs []float64) float64 {
	var sum float64
	for i, l := range lambdas {
		sum += Freshness(l, freqs[i])
	}
	return sum / float64(len(lambdas))
}

// UniformAllocation は1日の取り直しの予算 budget を全ページに同じ回数ずつ配る。
func UniformAllocation(lambdas []float64, budget float64) []float64 {
	out := make([]float64, len(lambdas))
	for i := range out {
		out[i] = budget / float64(len(lambdas))
	}
	return out
}

// ProportionalAllocation は budget を変化の頻度に比例させて配る。
func ProportionalAllocation(lambdas []float64, budget float64) []float64 {
	var total float64
	for _, l := range lambdas {
		total += l
	}
	out := make([]float64, len(lambdas))
	for i, l := range lambdas {
		out[i] = budget * l / total
	}
	return out
}

// OptimalAllocation は、回数の和が budget のもとで平均の鮮度を最大にする配分を返す（Cho と Garcia-Molina の方針）。
// ラグランジュの条件で、全ページの ∂F/∂f = g(λ/f)/λ を同じ値 μ にそろえる。g の上限は1なので、μλ が1以上のページ（変化が速すぎるページ）は0回になる。
func OptimalAllocation(lambdas []float64, budget float64) []float64 {
	if budget <= 0 || len(lambdas) == 0 {
		return make([]float64, len(lambdas))
	}
	minPositive := math.Inf(1)
	for _, l := range lambdas {
		if l > 0 {
			minPositive = min(minPositive, l)
		}
	}
	if math.IsInf(minPositive, 1) {
		return UniformAllocation(lambdas, budget)
	}
	allocate := func(mu float64, out []float64) float64 {
		var sum float64
		for i, l := range lambdas {
			out[i] = 0
			if l > 0 && mu*l < 1 {
				out[i] = l / inverseMarginal(mu*l)
			}
			sum += out[i]
		}
		return sum
	}
	atLo := make([]float64, len(lambdas))
	atHi := make([]float64, len(lambdas))
	hi := 1 / minPositive
	lo := hi / 2
	for allocate(lo, atLo) < budget {
		hi, lo = lo, lo/2
	}
	for range 200 {
		mid := math.Sqrt(lo * hi)
		if mid <= lo || mid >= hi {
			break
		}
		if allocate(mid, atHi) >= budget {
			lo = mid
		} else {
			hi = mid
		}
	}
	sumLo := allocate(lo, atLo)
	sumHi := allocate(hi, atHi)
	// μ が 1/λ に近づくと f は x の対数でしか0に近づかず、float64 の μ では λ/37 ほどから0へ飛ぶ。両端の配分を混ぜて和を budget に合わせる。
	w := (budget - sumHi) / (sumLo - sumHi)
	for i := range atHi {
		atHi[i] += w * (atLo[i] - atHi[i])
	}
	return atHi
}

// marginal は g(x) = 1 − (1+x)e^(−x)。x = λ/f のとき λ × ∂F/∂f にあたり、0から1へ単調に増える。
func marginal(x float64) float64 {
	return -math.Expm1(-x) - x*math.Exp(-x)
}

// inverseMarginal は 0 < y < 1 について g(x) = y となる x を二分法で返す。
func inverseMarginal(y float64) float64 {
	lo, hi := 0.0, 1.0
	for marginal(hi) < y {
		lo, hi = hi, 2*hi
	}
	for range 200 {
		mid := (lo + hi) / 2
		if mid <= lo || mid >= hi {
			break
		}
		if marginal(mid) < y {
			lo = mid
		} else {
			hi = mid
		}
	}
	return (lo + hi) / 2
}

// SimulateFreshness は、ページごとにポアソン過程の変化の時刻を引き、一定の間隔で intervals 回取り直したときに最新だった時間の割合を、全ページで平均して返す。
// ページ i の乱数は (seed, i) から作るので、ページの並びを変えても各ページの値は変わらない。0回のページは最新にならないので0とする。
func SimulateFreshness(lambdas, freqs []float64, intervals int, seed uint64) float64 {
	var sum float64
	for i, l := range lambdas {
		f := freqs[i]
		if f <= 0 {
			continue
		}
		if l <= 0 {
			sum++
			continue
		}
		rng := rand.New(rand.NewPCG(seed, uint64(i)))
		interval := 1 / f
		change := rng.ExpFloat64() / l
		var fresh float64
		for k := range intervals {
			fetched := float64(k) * interval
			for change < fetched {
				change += rng.ExpFloat64() / l
			}
			fresh += min(change-fetched, interval)
		}
		sum += fresh / (float64(intervals) * interval)
	}
	return sum / float64(len(lambdas))
}
