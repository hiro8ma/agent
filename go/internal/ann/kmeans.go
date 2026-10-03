package ann

import (
	"math"
	"math/rand/v2"
)

// sampleRows は n 行から size 行を種で決まる順に選び、詰め直した写しを返す。size が n 以上なら全行を使う。
func sampleRows(m Matrix, size int, r *rand.Rand) Matrix {
	if size >= m.N {
		return m
	}
	perm := r.Perm(m.N)[:size]
	data := make([]float32, size*m.Dim)
	for i, p := range perm {
		copy(data[i*m.Dim:], m.Row(p))
	}
	return Matrix{N: size, Dim: m.Dim, Data: data}
}

// kmeans は L2 の k-means（Lloyd 法）で k 個の重心を返す。初期値は k-means++ で、既に選んだ重心から遠い点ほど選ばれやすい。
// 割り当ては点ごとに独立なので並列にし、重心の更新は添字の順に足すので結果は並列度に依らない。
func kmeans(train Matrix, k, iters int, r *rand.Rand) Matrix {
	k = min(k, train.N)
	cents := seedPlusPlus(train, k, r)
	assign := make([]int32, train.N)
	sums := make([]float64, k*train.Dim)
	counts := make([]int, k)
	for range iters {
		assignAll(train, cents, assign)
		clear(sums)
		clear(counts)
		for i, c := range assign {
			counts[c]++
			row := train.Row(i)
			s := sums[int(c)*train.Dim:]
			for j, v := range row {
				s[j] += float64(v)
			}
		}
		for c := range k {
			dst := cents.Row(c)
			if counts[c] == 0 {
				copy(dst, train.Row(r.IntN(train.N)))
				continue
			}
			inv := 1 / float64(counts[c])
			for j := range dst {
				dst[j] = float32(sums[c*train.Dim+j] * inv)
			}
		}
	}
	return cents
}

func seedPlusPlus(train Matrix, k int, r *rand.Rand) Matrix {
	cents := Matrix{N: k, Dim: train.Dim, Data: make([]float32, k*train.Dim)}
	copy(cents.Row(0), train.Row(r.IntN(train.N)))
	d2 := make([]float64, train.N)
	for i := range d2 {
		d2[i] = math.Inf(1)
	}
	for c := 1; c < k; c++ {
		last := cents.Row(c - 1)
		parallelRange(train.N, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				d2[i] = min(d2[i], float64(l2sq(train.Row(i), last)))
			}
		})
		var total float64
		for _, d := range d2 {
			total += d
		}
		pick := r.IntN(train.N)
		if total > 0 {
			x := r.Float64() * total
			for i, d := range d2 {
				x -= d
				if x <= 0 {
					pick = i
					break
				}
			}
		}
		copy(cents.Row(c), train.Row(pick))
	}
	return cents
}

func assignAll(points, cents Matrix, out []int32) {
	parallelRange(points.N, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			out[i], _ = nearest(points.Row(i), cents)
		}
	})
}

func nearest(v []float32, cents Matrix) (int32, float32) {
	best, bestD := int32(0), float32(math.Inf(1))
	for c := range cents.N {
		if d := l2sq(v, cents.Row(c)); d < bestD {
			best, bestD = int32(c), d
		}
	}
	return best, bestD
}
