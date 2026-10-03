package ann

import (
	"fmt"
)

// PQConfig の値が 0 以下なら既定を使う。M は 8、K は 256、Iterations は 15、TrainSize は K×100。
type PQConfig struct {
	M, K       int
	Iterations int
	TrainSize  int
	Seed       uint64
}

func (c PQConfig) withDefaults() PQConfig {
	if c.M <= 0 {
		c.M = 8
	}
	if c.K <= 0 {
		c.K = 256
	}
	if c.Iterations <= 0 {
		c.Iterations = 15
	}
	if c.TrainSize <= 0 {
		c.TrainSize = c.K * 100
	}
	if c.Seed == 0 {
		c.Seed = 1
	}
	return c
}

// PQ は直積量子化器。Dim 次元を M 個のブロック（Sub 次元ずつ）に分け、ブロックごとに K 個の中心を持つ。
// ベクトルは各ブロックで最も近い中心の番号（1 バイト）の並びに置き換わり、M バイトになる。
type PQ struct {
	M, K, Dim, Sub int
	// Codebooks はブロック m の中心 c を [(m*K+c)*Sub, (m*K+c+1)*Sub) に持つ。
	Codebooks []float32
}

// NewPQ は与えた中心から量子化器を作る。学習済みの中心を読み込むときと、手で中心を決める例に使う。
func NewPQ(dim, m, k int, codebooks []float32) (*PQ, error) {
	if m <= 0 || dim%m != 0 {
		return nil, fmt.Errorf("ann: %d dimensions do not split into %d blocks", dim, m)
	}
	if k <= 0 || k > 256 {
		return nil, fmt.Errorf("ann: %d centroids per block, want 1..256 to fit a byte", k)
	}
	if len(codebooks) != m*k*(dim/m) {
		return nil, fmt.Errorf("ann: codebooks have %d values, want %d", len(codebooks), m*k*(dim/m))
	}
	return &PQ{M: m, K: k, Dim: dim, Sub: dim / m, Codebooks: codebooks}, nil
}

// TrainPQ はブロックごとに独立に k-means を回して中心を学習する。標本は Seed だけで決まる。
func TrainPQ(data Matrix, cfg PQConfig) (*PQ, error) {
	cfg = cfg.withDefaults()
	if data.Dim%cfg.M != 0 {
		return nil, fmt.Errorf("ann: %d dimensions do not split into %d blocks", data.Dim, cfg.M)
	}
	if cfg.K > 256 {
		return nil, fmt.Errorf("ann: %d centroids per block, want at most 256 to fit a byte", cfg.K)
	}
	r := newRand(cfg.Seed)
	train := sampleRows(data, cfg.TrainSize, r)
	sub := data.Dim / cfg.M
	k := min(cfg.K, train.N)
	books := make([]float32, cfg.M*k*sub)
	block := Matrix{N: train.N, Dim: sub, Data: make([]float32, train.N*sub)}
	for m := range cfg.M {
		for i := range train.N {
			copy(block.Row(i), train.Row(i)[m*sub:(m+1)*sub])
		}
		cents := kmeans(block, k, cfg.Iterations, r)
		copy(books[m*k*sub:], cents.Data)
	}
	return NewPQ(data.Dim, cfg.M, k, books)
}

func (p *PQ) centroid(m, c int) []float32 {
	off := (m*p.K + c) * p.Sub
	return p.Codebooks[off : off+p.Sub : off+p.Sub]
}

// Encode は v を M バイトの符号にする。
func (p *PQ) Encode(v []float32, code []uint8) {
	for m := range p.M {
		sub := v[m*p.Sub : (m+1)*p.Sub]
		best, bestD := 0, l2sq(sub, p.centroid(m, 0))
		for c := 1; c < p.K; c++ {
			if d := l2sq(sub, p.centroid(m, c)); d < bestD {
				best, bestD = c, d
			}
		}
		code[m] = uint8(best)
	}
}

// Decode は符号の中心をつないで Dim 次元の近似ベクトルに戻す。
func (p *PQ) Decode(code []uint8, out []float32) {
	for m, c := range code[:p.M] {
		copy(out[m*p.Sub:], p.centroid(m, int(c)))
	}
}

// Table は ADC（非対称距離計算）の距離表を作る。tab[m*K+c] はクエリのブロック m と中心 c の距離の 2 乗。
// クエリは量子化せずに使うので、近似の誤差はデータ側の量子化だけになる。
func (p *PQ) Table(q []float32, tab []float32) {
	for m := range p.M {
		sub := q[m*p.Sub : (m+1)*p.Sub]
		row := tab[m*p.K : (m+1)*p.K]
		for c := range row {
			row[c] = l2sq(sub, p.centroid(m, c))
		}
	}
}

// adcDistance は距離表を符号で引いて足す。M 回の表引きと加算で、元の Dim 次元の距離計算を置き換える。
func (p *PQ) adcDistance(tab []float32, code []uint8) float32 {
	var d float32
	for m, c := range code[:p.M] {
		d += tab[m*p.K+int(c)]
	}
	return d
}

// PQIndex は全件を PQ の符号だけで持ち、距離表を引いて全件を走査する。
type PQIndex struct {
	pq    *PQ
	n     int
	codes []uint8
}

func NewPQIndex(pq *PQ, data Matrix) *PQIndex {
	x := &PQIndex{pq: pq, n: data.N, codes: make([]uint8, data.N*pq.M)}
	parallelRange(data.N, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			pq.Encode(data.Row(i), x.codes[i*pq.M:(i+1)*pq.M])
		}
	})
	return x
}

func (x *PQIndex) BytesPerVector() int { return x.pq.M }

func (x *PQIndex) Code(i int) []uint8 { return x.codes[i*x.pq.M : (i+1)*x.pq.M] }

// Search は近似距離で近い k 件を返す。Dist は量子化した距離。
func (x *PQIndex) Search(q []float32, k int) []Neighbor {
	tab := make([]float32, x.pq.M*x.pq.K)
	x.pq.Table(q, tab)
	t := newKBest(min(k, x.n))
	m := x.pq.M
	for i := range x.n {
		t.push(Neighbor{ID: int32(i), Dist: x.pq.adcDistance(tab, x.codes[i*m:(i+1)*m])})
	}
	return t.sorted()
}

// SearchRefine は近似距離で candidates 件に絞り、元のベクトルで距離を取り直して k 件に並べ直す。
// 元のベクトルを別に持つ必要があるので、1 件あたりのメモリは符号だけのときより増える。
func (x *PQIndex) SearchRefine(q []float32, k, candidates int, e *Exact) []Neighbor {
	return e.Refine(q, ids(x.Search(q, max(k, candidates))), k)
}

func ids(ns []Neighbor) []int32 {
	out := make([]int32, len(ns))
	for i, n := range ns {
		out[i] = n.ID
	}
	return out
}
