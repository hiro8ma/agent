package ann

import (
	"math"
	"slices"
)

// IVFConfig の値が 0 以下なら既定を使う。NList は √N、NProbe は 1、Iterations は 10、TrainSize は NList×64。
type IVFConfig struct {
	NList      int
	NProbe     int
	Iterations int
	TrainSize  int
	Seed       uint64
}

func (c IVFConfig) withDefaults(n int) IVFConfig {
	if c.NList <= 0 {
		c.NList = max(1, int(math.Round(math.Sqrt(float64(n)))))
	}
	c.NList = min(c.NList, max(n, 1))
	if c.NProbe <= 0 {
		c.NProbe = 1
	}
	if c.Iterations <= 0 {
		c.Iterations = 10
	}
	if c.TrainSize <= 0 {
		c.TrainSize = c.NList * 64
	}
	if c.Seed == 0 {
		c.Seed = 1
	}
	return c
}

// SearchStats は 1 回の検索で距離を計算した件数。Centroids はクラスタの中心、Vectors は走査したベクトルの数。
type SearchStats struct {
	Centroids, Vectors int
}

// coarse は k-means の中心で全件を NList 個のクラスタに分け、クラスタごとに連続した添字の範囲を持つ。
type coarse struct {
	cents   Matrix
	offsets []int
	ids     []int32
}

// buildCoarse は中心を学習して全件を割り当て、元の添字から並べ替え後の位置への写像 order を返す。
func buildCoarse(data Matrix, cfg IVFConfig) (coarse, []int32) {
	r := newRand(cfg.Seed)
	cents := kmeans(sampleRows(data, cfg.TrainSize, r), cfg.NList, cfg.Iterations, r)
	assign := make([]int32, data.N)
	assignAll(data, cents, assign)
	c := coarse{cents: cents, offsets: make([]int, cents.N+1), ids: make([]int32, data.N)}
	for _, a := range assign {
		c.offsets[a+1]++
	}
	for i := range cents.N {
		c.offsets[i+1] += c.offsets[i]
	}
	next := slices.Clone(c.offsets[:cents.N])
	for i, a := range assign {
		c.ids[next[a]] = int32(i)
		next[a]++
	}
	return c, assign
}

// probe はクエリに近い nprobe 個のクラスタを近い順に返す。
func (c *coarse) probe(q []float32, nprobe int) []Neighbor {
	t := newKBest(min(nprobe, c.cents.N))
	for i := range c.cents.N {
		t.push(Neighbor{ID: int32(i), Dist: l2sq(q, c.cents.Row(i))})
	}
	return t.sorted()
}

func (c *coarse) NList() int { return c.cents.N }

func (c *coarse) ListSizes() []int {
	out := make([]int, c.cents.N)
	for i := range out {
		out[i] = c.offsets[i+1] - c.offsets[i]
	}
	return out
}

// IVFFlat はクラスタの中だけを元のベクトルで全探索する。ベクトルはクラスタごとに並べ替えて連続させる。
type IVFFlat struct {
	coarse
	nprobe int
	data   Matrix
	norms  []float32
}

func NewIVFFlat(data Matrix, cfg IVFConfig) *IVFFlat {
	cfg = cfg.withDefaults(data.N)
	c, _ := buildCoarse(data, cfg)
	ix := &IVFFlat{coarse: c, nprobe: cfg.NProbe, data: Matrix{N: data.N, Dim: data.Dim, Data: make([]float32, len(data.Data))}, norms: make([]float32, data.N)}
	for p, id := range c.ids {
		row := ix.data.Row(p)
		copy(row, data.Row(int(id)))
		ix.norms[p] = dot(row, row)
	}
	return ix
}

// WithNProbe は索引を共有したまま、走査するクラスタの数だけを変えた IVFFlat を返す。
func (ix *IVFFlat) WithNProbe(nprobe int) *IVFFlat {
	cp := *ix
	cp.nprobe = max(1, nprobe)
	return &cp
}

func (ix *IVFFlat) BytesPerVector() int { return 4*ix.data.Dim + 4 + 4 }

func (ix *IVFFlat) Search(q []float32, k int) []Neighbor {
	ns, _ := ix.SearchStats(q, k)
	return ns
}

func (ix *IVFFlat) SearchStats(q []float32, k int) ([]Neighbor, SearchStats) {
	lists := ix.probe(q, ix.nprobe)
	st := SearchStats{Centroids: ix.cents.N}
	t := newKBest(k)
	for _, l := range lists {
		lo, hi := ix.offsets[l.ID], ix.offsets[l.ID+1]
		st.Vectors += hi - lo
		for p := lo; p < hi; p++ {
			t.push(Neighbor{ID: ix.ids[p], Dist: ix.norms[p] - 2*dot(q, ix.data.Row(p))})
		}
	}
	qq := dot(q, q)
	out := t.sorted()
	for i := range out {
		out[i].Dist += qq
	}
	return out, st
}

// IVFPQConfig の EncodeRaw は残差ではなく元のベクトルを量子化する。残差を使う効果を測る比較用。
type IVFPQConfig struct {
	IVF       IVFConfig
	PQ        PQConfig
	EncodeRaw bool
}

// IVFPQ は各ベクトルから所属クラスタの中心を引いた残差を PQ で符号にする。量子化器は全クラスタで 1 つを共有する。
// 残差はクラスタの中心の周りに集まるので、元のベクトルより値の幅が狭く、同じ M と K でも量子化の誤差が小さい。
type IVFPQ struct {
	coarse
	nprobe int
	pq     *PQ
	codes  []uint8
	raw    bool
}

func NewIVFPQ(data Matrix, cfg IVFPQConfig) (*IVFPQ, error) {
	icfg := cfg.IVF.withDefaults(data.N)
	c, assign := buildCoarse(data, icfg)
	target := data
	if !cfg.EncodeRaw {
		target = residuals(data, c.cents, assign)
	}
	pq, err := TrainPQ(target, cfg.PQ)
	if err != nil {
		return nil, err
	}
	ix := &IVFPQ{coarse: c, nprobe: icfg.NProbe, pq: pq, codes: make([]uint8, data.N*pq.M), raw: cfg.EncodeRaw}
	parallelRange(data.N, func(lo, hi int) {
		for p := lo; p < hi; p++ {
			pq.Encode(target.Row(int(c.ids[p])), ix.codes[p*pq.M:(p+1)*pq.M])
		}
	})
	return ix, nil
}

func residuals(data, cents Matrix, assign []int32) Matrix {
	out := Matrix{N: data.N, Dim: data.Dim, Data: make([]float32, len(data.Data))}
	parallelRange(data.N, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			dst, src, c := out.Row(i), data.Row(i), cents.Row(int(assign[i]))
			for j := range dst {
				dst[j] = src[j] - c[j]
			}
		}
	})
	return out
}

func (ix *IVFPQ) WithNProbe(nprobe int) *IVFPQ {
	cp := *ix
	cp.nprobe = max(1, nprobe)
	return &cp
}

func (ix *IVFPQ) BytesPerVector() int { return ix.pq.M + 4 }

func (ix *IVFPQ) Search(q []float32, k int) []Neighbor {
	ns, _ := ix.SearchStats(q, k)
	return ns
}

// SearchStats は残差の版ではクラスタごとにクエリの残差 q-c で距離表を作り直す。元のベクトルの版は表を 1 回だけ作る。
func (ix *IVFPQ) SearchStats(q []float32, k int) ([]Neighbor, SearchStats) {
	lists := ix.probe(q, ix.nprobe)
	st := SearchStats{Centroids: ix.cents.N}
	m := ix.pq.M
	tab := make([]float32, m*ix.pq.K)
	rq := make([]float32, len(q))
	if ix.raw {
		ix.pq.Table(q, tab)
	}
	t := newKBest(k)
	for _, l := range lists {
		if !ix.raw {
			c := ix.cents.Row(int(l.ID))
			for j := range rq {
				rq[j] = q[j] - c[j]
			}
			ix.pq.Table(rq, tab)
		}
		lo, hi := ix.offsets[l.ID], ix.offsets[l.ID+1]
		st.Vectors += hi - lo
		for p := lo; p < hi; p++ {
			t.push(Neighbor{ID: ix.ids[p], Dist: ix.pq.adcDistance(tab, ix.codes[p*m:(p+1)*m])})
		}
	}
	return t.sorted(), st
}

func (ix *IVFPQ) SearchRefine(q []float32, k, candidates int, e *Exact) []Neighbor {
	return e.Refine(q, ids(ix.Search(q, max(k, candidates))), k)
}
