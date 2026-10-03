package index

import (
	"fmt"
	"math/rand/v2"
	"runtime"
	"slices"
	"sync"
	"testing"

	"github.com/hiro8ma/agent/go/simdsearch/vec"
)

// 教材の NumPy の実験と同じ形（N=100 万、d=128、K=20、正規乱数）。fp32 で 512MB。
const (
	knnN   = 1_000_000
	knnDim = 128
	knnK   = 20
	knnQ   = 32
)

// l2Index は ||q-x||² = ||q||² + ||x||² - 2q·x の ||x||² を先に求めて持つ。順位は 2q·x - ||x||² の大きい順で決まる。
type l2Index struct {
	*Index
	norms []float32
}

func newL2(ix *Index) *l2Index {
	l := &l2Index{Index: ix, norms: make([]float32, ix.N)}
	for i := range ix.N {
		l.norms[i] = vec.DotNaive(ix.Vec(i), ix.Vec(i))
	}
	return l
}

func (l *l2Index) search(q []float32, k int, dot DotFunc) []Result {
	t := newTopK(k)
	for id := range l.N {
		t.push(id, 2*dot(q, l.Vec(id))-l.norms[id])
	}
	return t.results()
}

func (l *l2Index) scanBatch(qs [][]float32, lo, hi int, dot DotFunc, tops []*topK) {
	for id := lo; id < hi; id++ {
		x, n := l.Vec(id), l.norms[id]
		for i, q := range qs {
			tops[i].push(id, 2*dot(q, x)-n)
		}
	}
}

// searchBatchParallel は行を CPU の数に分けてバッチで走査し、部分ごとの上位 k 件を併合する。
func (l *l2Index) searchBatchParallel(qs [][]float32, k int, dot DotFunc) [][]Result {
	workers := runtime.GOMAXPROCS(0)
	step := (l.N + workers - 1) / workers
	parts := make([][]*topK, workers)
	var wg sync.WaitGroup
	for w := range workers {
		wg.Go(func() {
			tops := make([]*topK, len(qs))
			for i := range tops {
				tops[i] = newTopK(k)
			}
			l.scanBatch(qs, w*step, min((w+1)*step, l.N), dot, tops)
			parts[w] = tops
		})
	}
	wg.Wait()
	out := make([][]Result, len(qs))
	for i := range qs {
		t := newTopK(k)
		for _, p := range parts {
			for _, r := range p[i].h {
				t.push(r.ID, r.Score)
			}
		}
		out[i] = t.results()
	}
	return out
}

func normalData(n, dim int, seed uint64) []float32 {
	r := rand.New(rand.NewPCG(seed, seed))
	out := make([]float32, n*dim)
	for i := range out {
		out[i] = float32(r.NormFloat64())
	}
	return out
}

func TestL2SearchAgreesAcrossDots(t *testing.T) {
	ix := newL2(New(normalData(5000, knnDim, 1), knnDim))
	qs := [][]float32{normalData(1, knnDim, 2), normalData(1, knnDim, 3)}
	par := ix.searchBatchParallel(qs, knnK, vec.DotSIMD8A)
	for i, q := range qs {
		want := ix.search(q, knnK, vec.DotNaive)
		if r := Recall(ix.search(q, knnK, vec.DotSIMD8A), want); r < 1 {
			t.Errorf("query %d: SIMD8A recall %.2f", i, r)
		}
		if r := Recall(par[i], want); r < 1 {
			t.Errorf("query %d: parallel batch recall %.2f", i, r)
		}
	}
}

var (
	knnOnce sync.Once
	knnIx   *l2Index
	knnQs   [][]float32
)

func knnSetup(b *testing.B) {
	b.Helper()
	knnOnce.Do(func() {
		knnIx = newL2(New(normalData(knnN, knnDim, 1), knnDim))
		for i := range knnQ {
			knnQs = append(knnQs, normalData(1, knnDim, uint64(100+i)))
		}
	})
}

func benchKNN(b *testing.B, dot DotFunc) {
	knnSetup(b)
	for b.Loop() {
		sink = knnIx.search(knnQs[0], knnK, dot)
	}
	b.ReportMetric(b.Elapsed().Seconds()*1e3/float64(b.N), "ms/query")
}

func BenchmarkKNN1MNaive(b *testing.B)   { benchKNN(b, vec.DotNaive) }
func BenchmarkKNN1MUnroll4(b *testing.B) { benchKNN(b, vec.DotUnroll4) }
func BenchmarkKNN1MSIMD8A(b *testing.B)  { benchKNN(b, vec.DotSIMD8A) }

func BenchmarkKNN1MBatch(b *testing.B) {
	for _, tc := range []struct {
		name     string
		n        int
		parallel bool
	}{{"B=32", knnQ, false}, {"B=1/parallel", 1, true}, {"B=32/parallel", knnQ, true}} {
		b.Run(tc.name, func(b *testing.B) {
			knnSetup(b)
			qs := knnQs[:tc.n]
			for b.Loop() {
				if tc.parallel {
					sinkBatch = knnIx.searchBatchParallel(qs, knnK, vec.DotSIMD8A)
					continue
				}
				tops := make([]*topK, len(qs))
				for i := range tops {
					tops[i] = newTopK(knnK)
				}
				knnIx.scanBatch(qs, 0, knnIx.N, vec.DotSIMD8A, tops)
			}
			b.ReportMetric(b.Elapsed().Seconds()*1e3/float64(b.N*tc.n), "ms/query")
		})
	}
}

// 256 ビット（uint64 × 4）の符号を 100 万件、ハミング距離で全走査する。
func BenchmarkHamming256Scan(b *testing.B) {
	const words = 4
	r := rand.New(rand.NewPCG(4, 4))
	codes := make([]uint64, knnN*words)
	for i := range codes {
		codes[i] = r.Uint64()
	}
	q := []uint64{r.Uint64(), r.Uint64(), r.Uint64(), r.Uint64()}
	for _, f := range []struct {
		name string
		ham  HammingFunc
	}{{"scalar", vec.Hamming}, {"simd", vec.HammingSIMD}} {
		b.Run(fmt.Sprint(f.name), func(b *testing.B) {
			for b.Loop() {
				t := newTopK(knnK)
				for id := range knnN {
					t.push(id, -float32(f.ham(q, codes[id*words:(id+1)*words:(id+1)*words])))
				}
				sink = slices.Clone(t.h)
			}
			b.ReportMetric(b.Elapsed().Seconds()*1e3/float64(b.N), "ms/query")
		})
	}
}
