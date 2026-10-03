// Package ann は近似最近傍探索（ANN）の主な方式を L2 距離で実装し、全探索と比べて再現率と速さを測る。
// 方式は全探索（ノルムの展開）/ 直積量子化（PQ）/ IVF / 局所性鋭敏型ハッシュ（LSH）/ HNSW。
package ann

import (
	"fmt"
	"math"
	"math/rand/v2"
	"runtime"
	"sync"
)

// Matrix は N 本の Dim 次元ベクトルを行ごとに 1 本の配列へ詰めて持つ。
type Matrix struct {
	N, Dim int
	Data   []float32
}

func NewMatrix(data []float32, dim int) (Matrix, error) {
	if dim <= 0 || len(data)%dim != 0 {
		return Matrix{}, fmt.Errorf("ann: %d values do not split into rows of %d", len(data), dim)
	}
	return Matrix{N: len(data) / dim, Dim: dim, Data: data}, nil
}

func (m Matrix) Row(i int) []float32 {
	return m.Data[i*m.Dim : (i+1)*m.Dim : (i+1)*m.Dim]
}

func newRand(seed uint64) *rand.Rand {
	return rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
}

// RandomNormal は各成分を標準正規分布から取る。近傍に構造が無く、ANN にとって最も不利な分布になる。
func RandomNormal(n, dim int, seed uint64) Matrix {
	r := newRand(seed)
	data := make([]float32, n*dim)
	for i := range data {
		data[i] = float32(r.NormFloat64())
	}
	return Matrix{N: n, Dim: dim, Data: data}
}

// GaussianMixture は clusters 個の中心の周りに点を散らし、長さ 1 に正規化する。
// noise は中心の長さ（約 1）に対する散らばりの大きさで、文の埋め込みのように塊がありつつ塊同士が重なる分布を作る。
// 同じ seed なら同じ中心を使うので、seed を共有して n だけ変えれば同じ分布からクエリを取り出せる。
func GaussianMixture(n, dim, clusters int, noise float64, seed, sampleSeed uint64) Matrix {
	r := newRand(seed)
	scale := 1 / math.Sqrt(float64(dim))
	cents := make([]float64, clusters*dim)
	for i := range cents {
		cents[i] = r.NormFloat64() * scale
	}
	s := newRand(sampleSeed)
	data := make([]float32, n*dim)
	row := make([]float64, dim)
	for i := range n {
		c := cents[s.IntN(clusters)*dim:]
		var norm float64
		for j := range dim {
			row[j] = c[j] + noise*s.NormFloat64()*scale
			norm += row[j] * row[j]
		}
		inv := 1 / math.Sqrt(norm)
		for j := range dim {
			data[i*dim+j] = float32(row[j] * inv)
		}
	}
	return Matrix{N: n, Dim: dim, Data: data}
}

func dot(a, b []float32) float32 {
	b = b[:len(a)]
	var s0, s1, s2, s3 float32
	i := 0
	for ; i+4 <= len(a); i += 4 {
		s0 += a[i] * b[i]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
	}
	for ; i < len(a); i++ {
		s0 += a[i] * b[i]
	}
	return (s0 + s1) + (s2 + s3)
}

func l2sq(a, b []float32) float32 {
	b = b[:len(a)]
	var s0, s1, s2, s3 float32
	i := 0
	for ; i+4 <= len(a); i += 4 {
		d0, d1, d2, d3 := a[i]-b[i], a[i+1]-b[i+1], a[i+2]-b[i+2], a[i+3]-b[i+3]
		s0 += d0 * d0
		s1 += d1 * d1
		s2 += d2 * d2
		s3 += d3 * d3
	}
	for ; i < len(a); i++ {
		d := a[i] - b[i]
		s0 += d * d
	}
	return (s0 + s1) + (s2 + s3)
}

// parallelRange は [0, n) を CPU の数に分けて fn を並列に呼ぶ。fn は添字ごとに独立した値だけを書くので、結果は分け方に依らない。
func parallelRange(n int, fn func(lo, hi int)) {
	workers := min(runtime.GOMAXPROCS(0), n)
	if workers <= 1 {
		fn(0, n)
		return
	}
	var wg sync.WaitGroup
	step := (n + workers - 1) / workers
	for lo := 0; lo < n; lo += step {
		wg.Go(func() { fn(lo, min(lo+step, n)) })
	}
	wg.Wait()
}

// Uniform は各成分を [0,1) の一様乱数から取る。次元が低いと近傍グラフの直径が N^(1/dim) で伸びるので、階層の効果が見える。
func Uniform(n, dim int, seed uint64) Matrix {
	r := newRand(seed)
	data := make([]float32, n*dim)
	for i := range data {
		data[i] = r.Float32()
	}
	return Matrix{N: n, Dim: dim, Data: data}
}
