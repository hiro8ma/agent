package ann

import (
	"cmp"
	"fmt"
	"math"
	"math/bits"
	"slices"
	"sync"
)

// Hamming は 2 つの符号で異なるビットの数。XOR で異なるビットだけを立て、64 ビットずつ立っているビットを数える。
func Hamming(a, b []uint64) int {
	b = b[:len(a)]
	n := 0
	for i := range a {
		n += bits.OnesCount64(a[i] ^ b[i])
	}
	return n
}

// LSH はランダムな超平面で空間を切り、各超平面のどちら側にあるかを 1 ビットにした符号を持つ（SimHash）。
// 2 本のベクトルの角度が θ のとき、1 枚の超平面で符号が分かれる確率は θ/π なので、ハミング距離が角度の推定になる。
type LSH struct {
	Bits, Words, Dim int
	planes           Matrix
	n                int
	sigs             []uint64
}

// NewLSH は bits 枚の超平面の法線を標準正規分布から取る。正規分布は向きが一様に散らばる。
func NewLSH(data Matrix, nbits int, seed uint64) (*LSH, error) {
	if nbits <= 0 {
		return nil, fmt.Errorf("ann: %d bits, want at least 1", nbits)
	}
	l := &LSH{Bits: nbits, Words: (nbits + 63) / 64, Dim: data.Dim, planes: RandomNormal(nbits, data.Dim, seed), n: data.N}
	l.sigs = make([]uint64, data.N*l.Words)
	parallelRange(data.N, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			l.Signature(data.Row(i), l.sigs[i*l.Words:(i+1)*l.Words])
		}
	})
	return l, nil
}

func (l *LSH) BytesPerVector() int { return 8 * l.Words }

// Signature は v を符号にする。ビット b は超平面 b の法線との内積が正なら 1。
func (l *LSH) Signature(v []float32, out []uint64) {
	clear(out)
	for b := range l.Bits {
		if dot(v, l.planes.Row(b)) > 0 {
			out[b/64] |= 1 << (b % 64)
		}
	}
}

func (l *LSH) sig(i int) []uint64 { return l.sigs[i*l.Words : (i+1)*l.Words : (i+1)*l.Words] }

// Candidates はクエリとハミング距離の近い順に c 件の ID を返す。距離は 0..Bits の整数なので、並べ替えずに距離ごとの件数で境目を決める。
func (l *LSH) Candidates(q []float32, c int) []int32 {
	qs := make([]uint64, l.Words)
	l.Signature(q, qs)
	dist := make([]uint16, l.n)
	hist := make([]int, l.Bits+1)
	for i := range l.n {
		d := Hamming(qs, l.sig(i))
		dist[i] = uint16(d)
		hist[d]++
	}
	c = min(c, l.n)
	limit, below := 0, 0
	for limit <= l.Bits && below+hist[limit] < c {
		below += hist[limit]
		limit++
	}
	out := make([]int32, 0, c)
	tie := c - below
	for i, d := range dist {
		switch {
		case int(d) < limit:
			out = append(out, int32(i))
		case int(d) == limit && tie > 0:
			out = append(out, int32(i))
			tie--
		}
	}
	return out
}

// Within はハミング距離が radius 以下の ID を返す。件数はクエリによって変わる。
func (l *LSH) Within(q []float32, radius int) []int32 {
	qs := make([]uint64, l.Words)
	l.Signature(q, qs)
	var out []int32
	for i := range l.n {
		if Hamming(qs, l.sig(i)) <= radius {
			out = append(out, int32(i))
		}
	}
	return out
}

// Search はハミング距離で c 件に絞り、元のベクトルで並べ直して k 件を返す。
func (l *LSH) Search(q []float32, k, c int, e *Exact) []Neighbor {
	return e.Refine(q, l.Candidates(q, c), k)
}

// NewLSHPlanes は法線を与えて符号を作る。単位行列を渡すと各成分の符号をそのまま 1 ビットにする二値化になる。
func NewLSHPlanes(data Matrix, planes Matrix) (*LSH, error) {
	if planes.Dim != data.Dim || planes.N == 0 {
		return nil, fmt.Errorf("ann: planes are %dx%d, want rows of %d dimensions", planes.N, planes.Dim, data.Dim)
	}
	l := &LSH{Bits: planes.N, Words: (planes.N + 63) / 64, Dim: data.Dim, planes: planes, n: data.N}
	l.sigs = make([]uint64, data.N*l.Words)
	parallelRange(data.N, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			l.Signature(data.Row(i), l.sigs[i*l.Words:(i+1)*l.Words])
		}
	})
	return l, nil
}

// Identity は dim 次元の単位行列。
func Identity(dim int) Matrix {
	m := Matrix{N: dim, Dim: dim, Data: make([]float32, dim*dim)}
	for i := range dim {
		m.Data[i*dim+i] = 1
	}
	return m
}

// LSHIndexConfig の Bits は 1 つの表のバケット番号のビット数で、表は 2^Bits 個のバケットを持つ。
// Probes は 1 つの表で探すバケットの数で、クエリのバケットにハミング距離 1 の隣を足していく（1..Bits+1）。
// MinCandidates は候補がこの数に満たない間、Probes を超えて隣のバケットを足す。
type LSHIndexConfig struct {
	Bits, Tables  int
	Probes        int
	MinCandidates int
	Seed          uint64
}

// LSHIndex は SimHash の Bits ビットをバケットの番号にした転置ファイル。表ごとに別の超平面を使い、候補を和集合にする。
type LSHIndex struct {
	cfg     LSHIndexConfig
	n       int
	planes  []Matrix
	offsets [][]int
	ids     [][]int32
	pool    *sync.Pool
}

func NewLSHIndex(data Matrix, cfg LSHIndexConfig) (*LSHIndex, error) {
	if cfg.Bits <= 0 || cfg.Bits > 24 {
		return nil, fmt.Errorf("ann: %d bucket bits, want 1..24", cfg.Bits)
	}
	cfg.Tables = max(cfg.Tables, 1)
	cfg.Probes = min(max(cfg.Probes, 1), cfg.Bits+1)
	if cfg.Seed == 0 {
		cfg.Seed = 1
	}
	n := data.N
	x := &LSHIndex{cfg: cfg, n: n, pool: &sync.Pool{New: func() any { return &visited{marks: make([]uint32, n)} }}}
	nb := 1 << cfg.Bits
	keys := make([]uint32, n)
	for t := range cfg.Tables {
		planes := RandomNormal(cfg.Bits, data.Dim, cfg.Seed+uint64(t))
		parallelRange(n, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				keys[i], _ = bucketKey(planes, data.Row(i), nil)
			}
		})
		offsets := make([]int, nb+1)
		for _, k := range keys {
			offsets[k+1]++
		}
		for b := range nb {
			offsets[b+1] += offsets[b]
		}
		next := slices.Clone(offsets[:nb])
		ids := make([]int32, n)
		for i, k := range keys {
			ids[next[k]] = int32(i)
			next[k]++
		}
		x.planes = append(x.planes, planes)
		x.offsets = append(x.offsets, offsets)
		x.ids = append(x.ids, ids)
	}
	return x, nil
}

// bucketKey は超平面の側をビットにしたバケット番号を返す。margins があれば各超平面との内積の絶対値を書く。
func bucketKey(planes Matrix, v []float32, margins []float32) (uint32, []float32) {
	var key uint32
	for b := range planes.N {
		p := dot(v, planes.Row(b))
		if p >= 0 {
			key |= 1 << b
		}
		if margins != nil {
			margins[b] = float32(math.Abs(float64(p)))
		}
	}
	return key, margins
}

func (x *LSHIndex) WithProbes(probes, minCandidates int) *LSHIndex {
	cp := *x
	cp.cfg.Probes = min(max(probes, 1), x.cfg.Bits+1)
	cp.cfg.MinCandidates = minCandidates
	return &cp
}

func (x *LSHIndex) BytesPerVector() int { return 4 * x.cfg.Tables }

// Candidates は各表でクエリのバケットと、超平面に近い（符号が反転しやすい）ビットから順に反転した隣のバケットを探す。
// 戻り値の buckets は探したバケットの数。
func (x *LSHIndex) Candidates(q []float32) (out []int32, buckets int) {
	v := x.pool.Get().(*visited)
	defer x.pool.Put(v)
	v.reset()
	b := x.cfg.Bits
	margins := make([]float32, b)
	order := make([]int, b)
	for t, planes := range x.planes {
		key, _ := bucketKey(planes, q, margins)
		for i := range order {
			order[i] = i
		}
		slices.SortStableFunc(order, func(i, j int) int { return cmp.Compare(margins[i], margins[j]) })
		for p := 0; p <= b; p++ {
			if p >= x.cfg.Probes && len(out) >= x.cfg.MinCandidates {
				break
			}
			k := key
			if p > 0 {
				k ^= 1 << order[p-1]
			}
			buckets++
			for _, id := range x.ids[t][x.offsets[t][k]:x.offsets[t][k+1]] {
				if v.visit(id) {
					out = append(out, id)
				}
			}
		}
	}
	return out, buckets
}

// Search は候補を元のベクトルで並べ直す。Stats の Vectors は距離を計算した候補の数。
func (x *LSHIndex) Search(q []float32, k int, e *Exact) ([]Neighbor, SearchStats) {
	c, _ := x.Candidates(q)
	return e.Refine(q, c, k), SearchStats{Vectors: len(c)}
}
