package search

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"
	"runtime"
	"sync"
)

const (
	DefaultIVFIterations = 10
	// DefaultIVFTrainPerList は k-means の学習に使う 1 グループあたりの点の数。全件で学習すると構築が N×nlist に比例して重くなるため標本で足りる分に抑える。
	DefaultIVFTrainPerList = 64
	DefaultIVFSeed         = 1
)

// IVFConfig の値が 0 以下なら既定を使う。NList は √N、NProbe は 1、TrainSize は NList×DefaultIVFTrainPerList。
type IVFConfig struct {
	NList      int
	NProbe     int
	Iterations int
	TrainSize  int
	Seed       uint64
}

// IVF は文書のベクトルを k-means で NList 個のグループに分け、クエリに近い NProbe 個のグループの中だけを比べる近似の最近傍探索。
// ベクトルはグループごとに連続して並べ、走査で読むメモリを連続させる。
type IVF struct {
	docs      []Doc
	embedder  Embedder
	nprobe    int
	centroids vectorSet
	vecs      vectorSet
	ids       []int32
	offsets   []int
}

var _ Retriever = (*IVF)(nil)

// NewIVF は同じ入力と設定なら同じ索引を作る。k-means の初期値と学習の標本は Seed だけで決まる。
func NewIVF(docs []Doc, vecs [][]float32, e Embedder, cfg IVFConfig) (*IVF, error) {
	if len(docs) != len(vecs) {
		return nil, fmt.Errorf("search: %d docs but %d vectors", len(docs), len(vecs))
	}
	all, err := newVectorSet(vecs)
	if err != nil {
		return nil, err
	}
	n := all.len()
	cfg = cfg.withDefaults(n)
	ix := &IVF{docs: docs, embedder: e, nprobe: cfg.NProbe, vecs: vectorSet{dim: all.dim}}
	if n == 0 {
		ix.offsets = []int{0}
		return ix, nil
	}
	ix.centroids = trainKMeans(all, cfg)
	assign := assignNearest(all, ix.centroids)

	nlist := ix.centroids.len()
	ix.offsets = make([]int, nlist+1)
	for _, c := range assign {
		ix.offsets[c+1]++
	}
	for c := range nlist {
		ix.offsets[c+1] += ix.offsets[c]
	}
	next := make([]int, nlist)
	copy(next, ix.offsets[:nlist])
	ix.vecs.data = make([]float32, len(all.data))
	ix.ids = make([]int32, n)
	for i, c := range assign {
		p := next[c]
		next[c]++
		copy(ix.vecs.row(p), all.row(i))
		ix.ids[p] = int32(i)
	}
	return ix, nil
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
		c.Iterations = DefaultIVFIterations
	}
	if c.TrainSize <= 0 {
		c.TrainSize = c.NList * DefaultIVFTrainPerList
	}
	c.TrainSize = min(max(c.TrainSize, c.NList), n)
	if c.Seed == 0 {
		c.Seed = DefaultIVFSeed
	}
	return c
}

// WithNProbe は索引を共有したまま、走査するグループの数だけを変えた IVF を返す。
func (ix *IVF) WithNProbe(nprobe int) *IVF {
	cp := *ix
	cp.nprobe = max(1, nprobe)
	return &cp
}

func (ix *IVF) NList() int { return ix.centroids.len() }

func (ix *IVF) ListSizes() []int {
	sizes := make([]int, ix.NList())
	for c := range sizes {
		sizes[c] = ix.offsets[c+1] - ix.offsets[c]
	}
	return sizes
}

func (ix *IVF) SearchVector(q []float32, limit int) ([]VectorHit, error) {
	q, err := ix.vecs.query(q)
	if err != nil {
		return nil, err
	}
	nlist := ix.NList()
	probe := newTopK(min(ix.nprobe, nlist))
	for c := range nlist {
		probe.push(VectorHit{ID: c, Score: dot(q, ix.centroids.row(c))})
	}
	top := newTopK(resultSize(limit, len(ix.ids)))
	for _, p := range probe.heap {
		for i := ix.offsets[p.ID]; i < ix.offsets[p.ID+1]; i++ {
			top.push(VectorHit{ID: int(ix.ids[i]), Score: dot(q, ix.vecs.row(i))})
		}
	}
	return top.sorted(), nil
}

func (ix *IVF) Search(ctx context.Context, query string, limit int) ([]Doc, error) {
	q, err := embedQuery(ctx, ix.embedder, query)
	if err != nil {
		return nil, err
	}
	hits, err := ix.SearchVector(q, limit)
	if err != nil {
		return nil, err
	}
	return hitsToDocs(ix.docs, hits), nil
}

// seedPlusPlus は k-means++ で初期の重心を選ぶ。既に選んだ重心から遠い点ほど選ばれやすく、同じ塊に 2 つの重心が置かれて抜け出せなくなるのを防ぐ。
func seedPlusPlus(train vectorSet, k int, r *rand.Rand) vectorSet {
	n := train.len()
	cents := vectorSet{dim: train.dim, data: make([]float32, 0, k*train.dim)}
	dist := make([]float64, n)
	for i := range dist {
		dist[i] = math.Inf(1)
	}
	next := r.IntN(n)
	for len(cents.data) < k*train.dim {
		cents.data = append(cents.data, train.row(next)...)
		c := cents.row(cents.len() - 1)
		parallelRange(n, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				d := max(0, 1-float64(dot(train.row(i), c)))
				dist[i] = min(dist[i], d*d)
			}
		})
		total := 0.0
		for _, d := range dist {
			total += d
		}
		if total == 0 {
			next = r.IntN(n)
			continue
		}
		x := r.Float64() * total
		next = n - 1
		for i, d := range dist {
			if x -= d; x < 0 {
				next = i
				break
			}
		}
	}
	return cents
}

// trainKMeans は球面 k-means（内積で割り当て、重心を正規化し直す）。重心が空になったら標本から種で決まる点を選び直す。
func trainKMeans(all vectorSet, cfg IVFConfig) vectorSet {
	r := rand.New(rand.NewPCG(cfg.Seed, uint64(all.len())))
	perm := r.Perm(all.len())
	train := vectorSet{dim: all.dim, data: make([]float32, 0, cfg.TrainSize*all.dim)}
	for _, i := range perm[:cfg.TrainSize] {
		train.data = append(train.data, all.row(i)...)
	}
	cents := seedPlusPlus(train, cfg.NList, r)

	sums := make([]float64, len(cents.data))
	counts := make([]int, cfg.NList)
	for range cfg.Iterations {
		assign := assignNearest(train, cents)
		clear(sums)
		clear(counts)
		for i, c := range assign {
			counts[c]++
			s := sums[c*all.dim : (c+1)*all.dim]
			for j, x := range train.row(i) {
				s[j] += float64(x)
			}
		}
		for c := range cfg.NList {
			row := cents.row(c)
			if counts[c] == 0 {
				copy(row, train.row(r.IntN(train.len())))
				continue
			}
			for j := range row {
				row[j] = float32(sums[c*all.dim+j])
			}
			normalize(row)
		}
	}
	return cents
}

// assignNearest は各点に内積が最大の重心を割り当てる。点ごとに独立なので並列にしても結果は変わらない。
func assignNearest(points, cents vectorSet) []int {
	out := make([]int, points.len())
	parallelRange(points.len(), func(lo, hi int) {
		for i := lo; i < hi; i++ {
			p := points.row(i)
			best, bestSim := 0, float32(math.Inf(-1))
			for c := range cents.len() {
				if s := dot(p, cents.row(c)); s > bestSim {
					best, bestSim = c, s
				}
			}
			out[i] = best
		}
	})
	return out
}

// parallelRange は [0, n) を CPU の数に分けて fn を並列に呼ぶ。fn は添字ごとに独立した値だけを書く。
func parallelRange(n int, fn func(lo, hi int)) {
	if n == 0 {
		return
	}
	workers := min(runtime.GOMAXPROCS(0), n)
	chunk := (n + workers - 1) / workers
	var wg sync.WaitGroup
	for lo := 0; lo < n; lo += chunk {
		wg.Go(func() { fn(lo, min(lo+chunk, n)) })
	}
	wg.Wait()
}
