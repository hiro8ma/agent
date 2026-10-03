package ann

import (
	"math"
	"slices"
	"sync"
)

// HNSWConfig の値が 0 以下なら既定を使う。M は 16、EfConstruction は 200、EfSearch は 64。
// SimpleNeighbors は近い順に M 本をつなぐ。偽なら多様性を見るヒューリスティックで選ぶ。
// Flat は全点を層 0 にだけ置き、階層を持たない NSW（navigable small world）のグラフにする。
type HNSWConfig struct {
	M               int
	EfConstruction  int
	EfSearch        int
	Seed            uint64
	SimpleNeighbors bool
	Flat            bool
}

// HNSWStats の Evals は距離を計算した点の数、Hops はたどった辺の数（貪欲探索では移動の回数、ef の探索では隣を読んだ点の数）。
type HNSWStats struct {
	Evals, Hops int
}

func (c HNSWConfig) withDefaults() HNSWConfig {
	if c.M <= 0 {
		c.M = 16
	}
	if c.EfConstruction <= 0 {
		c.EfConstruction = 200
	}
	if c.EfSearch <= 0 {
		c.EfSearch = 64
	}
	if c.Seed == 0 {
		c.Seed = 1
	}
	return c
}

// hnswMaxLevel は層の上限。M=16 で層 l に上がる確率は 16^-l なので、10 億件でも 8 層に届かない。
const hnswMaxLevel = 16

// HNSW は階層つきの近傍グラフ。上の層ほど点が少なく辺が長いので、上の層で大まかに近づき、層 0 で細かく探す。
type HNSW struct {
	data     Matrix
	m, m0    int
	efC      int
	efSearch int
	simple   bool
	levels   []uint8
	// layer0 は点ごとに m0+1 個の枠を持ち、先頭が辺の数、続く枠が隣の ID。
	layer0 []int32
	// upper[id] は層 1..levels[id] の枠を m+1 個ずつ並べる。
	upper    [][]int32
	entry    int32
	maxLevel int
	pool     sync.Pool
}

// sampleLevels は各点の最上層を -ln(U)/ln(M) の切り捨てで決める。層 l 以上になる確率は M^-l の幾何分布になる。
func sampleLevels(n, m int, seed uint64) []uint8 {
	r := newRand(seed)
	mL := 1 / math.Log(float64(m))
	out := make([]uint8, n)
	for i := range out {
		out[i] = uint8(min(int(-math.Log(1-r.Float64())*mL), hnswMaxLevel))
	}
	return out
}

type visited struct {
	stamp uint32
	marks []uint32
}

func (v *visited) reset() {
	v.stamp++
	if v.stamp == 0 {
		clear(v.marks)
		v.stamp = 1
	}
}

func (v *visited) visit(id int32) bool {
	if v.marks[id] == v.stamp {
		return false
	}
	v.marks[id] = v.stamp
	return true
}

// NewHNSW は data の行を 0 から順に挿入する。層の高さは Seed で決まる乱数で先に決め、同じ入力と設定なら同じグラフを作る。
// data は写さずに参照するので、索引を使う間は書き換えないこと。
func NewHNSW(data Matrix, cfg HNSWConfig) *HNSW {
	cfg = cfg.withDefaults()
	h := &HNSW{
		data: data, m: cfg.M, m0: 2 * cfg.M, efC: cfg.EfConstruction, efSearch: cfg.EfSearch, simple: cfg.SimpleNeighbors,
		levels: make([]uint8, data.N), upper: make([][]int32, data.N),
	}
	h.layer0 = make([]int32, data.N*(h.m0+1))
	h.pool.New = func() any { return &visited{marks: make([]uint32, data.N)} }
	if !cfg.Flat {
		h.levels = sampleLevels(data.N, cfg.M, cfg.Seed)
	}
	for i, l := range h.levels {
		if l > 0 {
			h.upper[i] = make([]int32, int(l)*(h.m+1))
		}
	}
	v := h.pool.Get().(*visited)
	defer h.pool.Put(v)
	for i := range data.N {
		h.insert(int32(i), v)
	}
	return h
}

// WithEfSearch はグラフを共有したまま、層 0 で持つ候補の数だけを変えた HNSW を返す。
func (h *HNSW) WithEfSearch(ef int) *HNSW {
	return &HNSW{
		data: h.data, m: h.m, m0: h.m0, efC: h.efC, efSearch: max(1, ef), simple: h.simple,
		levels: h.levels, layer0: h.layer0, upper: h.upper, entry: h.entry, maxLevel: h.maxLevel,
		pool: sync.Pool{New: h.pool.New},
	}
}

func (h *HNSW) slot(id int32, lc int) []int32 {
	if lc == 0 {
		off := int(id) * (h.m0 + 1)
		return h.layer0[off : off+h.m0+1]
	}
	off := (lc - 1) * (h.m + 1)
	return h.upper[id][off : off+h.m+1]
}

func (h *HNSW) neighbors(id int32, lc int) []int32 {
	s := h.slot(id, lc)
	return s[1 : 1+s[0]]
}

func (h *HNSW) setNeighbors(id int32, lc int, ns []Neighbor) {
	s := h.slot(id, lc)
	s[0] = int32(len(ns))
	for i, n := range ns {
		s[1+i] = n.ID
	}
}

func (h *HNSW) dist(q []float32, id int32) float32 { return l2sq(q, h.data.Row(int(id))) }

// searchLayer は層 lc で、入口から近い ef 件を貪欲に広げて探す。最も近い未展開の候補が持っている ef 件の最遠より遠くなったら止める。
func (h *HNSW) searchLayer(q []float32, eps []Neighbor, ef, lc int, v *visited, st *HNSWStats) []Neighbor {
	cand := make(minQueue, 0, ef)
	res := newKBest(ef)
	for _, ep := range eps {
		if v.visit(ep.ID) {
			cand.push(ep)
			res.push(ep)
		}
	}
	for len(cand) > 0 {
		c := cand.pop()
		if w, ok := res.worst(); ok && closer(w, c) {
			break
		}
		st.Hops++
		for _, e := range h.neighbors(c.ID, lc) {
			if !v.visit(e) {
				continue
			}
			st.Evals++
			n := Neighbor{ID: e, Dist: h.dist(q, e)}
			if w, ok := res.worst(); !ok || closer(n, w) {
				cand.push(n)
				res.push(n)
			}
		}
	}
	return res.sorted()
}

// greedyLayer は候補を 1 つだけ持ち、隣に今より近い点がある限り最も近い隣へ移る。極小に着いたら止まる。
func (h *HNSW) greedyLayer(q []float32, ep Neighbor, lc int, st *HNSWStats) Neighbor {
	for {
		best := ep
		for _, e := range h.neighbors(ep.ID, lc) {
			st.Evals++
			if n := (Neighbor{ID: e, Dist: h.dist(q, e)}); closer(n, best) {
				best = n
			}
		}
		if best == ep {
			return ep
		}
		ep = best
		st.Hops++
	}
}

// descend は入口から層 1 まで貪欲に降りて、層 0 の入口を返す。
func (h *HNSW) descend(q []float32, st *HNSWStats) Neighbor {
	ep := Neighbor{ID: h.entry, Dist: h.dist(q, h.entry)}
	st.Evals++
	for lc := h.maxLevel; lc > 0; lc-- {
		ep = h.greedyLayer(q, ep, lc, st)
	}
	return ep
}

// selectNeighbors は近い順に並んだ候補から m 本を選ぶ。ヒューリスティックは、既に選んだ点の方が候補に近ければ候補を捨てる。
// 同じ塊の点ばかりをつなぐと塊の外へ出る辺が無くなり、離れた塊へたどり着けなくなるのを防ぐ。
func (h *HNSW) selectNeighbors(cands []Neighbor, m int) []Neighbor {
	if h.simple || len(cands) <= m {
		return cands[:min(m, len(cands))]
	}
	out := make([]Neighbor, 0, m)
	for _, c := range cands {
		row := h.data.Row(int(c.ID))
		keep := true
		for _, r := range out {
			if h.dist(row, r.ID) < c.Dist {
				keep = false
				break
			}
		}
		if keep {
			out = append(out, c)
			if len(out) == m {
				break
			}
		}
	}
	return out
}

func (h *HNSW) insert(id int32, v *visited) {
	level := int(h.levels[id])
	if id == 0 {
		h.entry, h.maxLevel = 0, level
		return
	}
	q := h.data.Row(int(id))
	var st HNSWStats
	ep := Neighbor{ID: h.entry, Dist: h.dist(q, h.entry)}
	for lc := h.maxLevel; lc > level; lc-- {
		ep = h.greedyLayer(q, ep, lc, &st)
	}
	eps := []Neighbor{ep}
	for lc := min(level, h.maxLevel); lc >= 0; lc-- {
		v.reset()
		w := h.searchLayer(q, eps, h.efC, lc, v, &st)
		chosen := h.selectNeighbors(w, h.m)
		h.setNeighbors(id, lc, chosen)
		maxConn := h.m
		if lc == 0 {
			maxConn = h.m0
		}
		for _, n := range chosen {
			h.link(n.ID, id, n.Dist, lc, maxConn)
		}
		eps = w
	}
	if level > h.maxLevel {
		h.entry, h.maxLevel = id, level
	}
}

// link は from の辺に to を足す。枠が埋まっていれば、既存の辺と to を合わせて選び直す。
func (h *HNSW) link(from, to int32, d float32, lc, maxConn int) {
	s := h.slot(from, lc)
	if int(s[0]) < maxConn {
		s[1+s[0]] = to
		s[0]++
		return
	}
	base := h.data.Row(int(from))
	cands := make([]Neighbor, 0, maxConn+1)
	cands = append(cands, Neighbor{ID: to, Dist: d})
	for _, e := range s[1 : 1+s[0]] {
		cands = append(cands, Neighbor{ID: e, Dist: h.dist(base, e)})
	}
	slices.SortFunc(cands, compareNeighbor)
	h.setNeighbors(from, lc, h.selectNeighbors(cands, maxConn))
}

func (h *HNSW) Search(q []float32, k int) []Neighbor {
	ns, _ := h.SearchStats(q, k)
	return ns
}

// SearchStats は上の層を貪欲に降り、層 0 で max(EfSearch, k) 件の候補を持って探す。
func (h *HNSW) SearchStats(q []float32, k int) ([]Neighbor, HNSWStats) {
	var st HNSWStats
	if h.data.N == 0 {
		return nil, st
	}
	ep := h.descend(q, &st)
	v := h.pool.Get().(*visited)
	defer h.pool.Put(v)
	v.reset()
	res := h.searchLayer(q, []Neighbor{ep}, max(h.efSearch, k), 0, v, &st)
	return res[:min(k, len(res))], st
}

// Greedy は層 0 でも候補を 1 つだけ持つ貪欲探索で、着いた極小の 1 点を返す。
func (h *HNSW) Greedy(q []float32) (Neighbor, HNSWStats) {
	var st HNSWStats
	if h.data.N == 0 {
		return Neighbor{}, st
	}
	ep := h.descend(q, &st)
	return h.greedyLayer(q, ep, 0, &st), st
}

// LevelCounts は最上層が l の点の数を l ごとに返す。層 l に上がる確率が M^-l なので、件数は層ごとに約 1/M に減る。
func (h *HNSW) LevelCounts() []int {
	out := make([]int, h.maxLevel+1)
	for _, l := range h.levels {
		out[l]++
	}
	return out
}

// Bytes はグラフの辺の枠と層の高さが使うバイト数。ベクトル本体は含めない。
func (h *HNSW) Bytes() int {
	n := 4*len(h.layer0) + len(h.levels) + 24*len(h.upper)
	for _, u := range h.upper {
		n += 4 * len(u)
	}
	return n
}

// BytesPerVector はベクトル本体（fp32）とグラフを合わせた 1 件あたりのバイト数。
func (h *HNSW) BytesPerVector() float64 {
	if h.data.N == 0 {
		return 0
	}
	return float64(4*h.data.Dim) + float64(h.Bytes())/float64(h.data.N)
}

func (h *HNSW) MaxLevel() int { return h.maxLevel }
