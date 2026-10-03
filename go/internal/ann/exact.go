package ann

import (
	"fmt"
	"slices"
	"sync"
)

// Exact は全件と距離を取る厳密な K 近傍探索。||q-x||² = ||q||² + ||x||² - 2q·x と展開し、
// ||x||² は構築時に求めておく。||q||² はどの x でも同じなので順位付けでは足さず、返す直前にだけ足す。
type Exact struct {
	m     Matrix
	norms []float32
}

func NewExact(m Matrix) *Exact {
	norms := make([]float32, m.N)
	for i := range m.N {
		r := m.Row(i)
		norms[i] = dot(r, r)
	}
	return &Exact{m: m, norms: norms}
}

func (e *Exact) Len() int { return e.m.N }

func (e *Exact) BytesPerVector() int { return 4*e.m.Dim + 4 }

func (e *Exact) check(q []float32) {
	if len(q) != e.m.Dim {
		panic(fmt.Sprintf("ann: query has %d dimensions, want %d", len(q), e.m.Dim))
	}
}

// Search はヒープで k 件だけを持ち、近い順に並べて返す。
func (e *Exact) Search(q []float32, k int) []Neighbor {
	e.check(q)
	t := newKBest(min(k, e.m.N))
	e.scan(q, 0, e.m.N, t)
	return e.finish(q, t.sorted())
}

func (e *Exact) scan(q []float32, lo, hi int, t *kBest) {
	for i := lo; i < hi; i++ {
		t.push(Neighbor{ID: int32(i), Dist: e.norms[i] - 2*dot(q, e.m.Row(i))})
	}
}

func (e *Exact) finish(q []float32, ns []Neighbor) []Neighbor {
	qq := dot(q, q)
	for i := range ns {
		ns[i].Dist += qq
	}
	return ns
}

// Partition は全件の距離を buf に書き、選択アルゴリズムで近い k 件を先頭に集める。順序は付けない（argpartition 相当）。
// buf は長さ N 以上なら使い回し、足りなければ確保する。
func (e *Exact) Partition(q []float32, k int, buf []Neighbor) []Neighbor {
	e.check(q)
	if cap(buf) < e.m.N {
		buf = make([]Neighbor, e.m.N)
	}
	buf = buf[:e.m.N]
	for i := range e.m.N {
		buf[i] = Neighbor{ID: int32(i), Dist: e.norms[i] - 2*dot(q, e.m.Row(i))}
	}
	k = min(k, e.m.N)
	selectK(buf, k)
	return e.finish(q, slices.Clone(buf[:k]))
}

// exactBlock は SearchBatch で 1 度に読む行数。128 次元の fp32 で 64KB になり、L1 に載ったまま全クエリと掛け合わせられる。
const exactBlock = 128

// SearchBatch は行をブロックに分け、1 ブロックを読むたびに全クエリと内積を取る。DB を読む回数がクエリ数によらず 1 回になる。
func (e *Exact) SearchBatch(qs [][]float32, k int) [][]Neighbor {
	tops := make([]*kBest, len(qs))
	for i, q := range qs {
		e.check(q)
		tops[i] = newKBest(min(k, e.m.N))
	}
	e.scanBatch(qs, 0, e.m.N, tops)
	out := make([][]Neighbor, len(qs))
	for i, t := range tops {
		out[i] = e.finish(qs[i], t.sorted())
	}
	return out
}

func (e *Exact) scanBatch(qs [][]float32, lo, hi int, tops []*kBest) {
	for b := lo; b < hi; b += exactBlock {
		end := min(b+exactBlock, hi)
		qi := 0
		for ; qi+4 <= len(qs); qi += 4 {
			q0, q1, q2, q3 := qs[qi], qs[qi+1], qs[qi+2], qs[qi+3]
			t0, t1, t2, t3 := tops[qi], tops[qi+1], tops[qi+2], tops[qi+3]
			for i := b; i < end; i++ {
				d0, d1, d2, d3 := dot4(e.m.Row(i), q0, q1, q2, q3)
				n := e.norms[i]
				id := int32(i)
				t0.push(Neighbor{ID: id, Dist: n - 2*d0})
				t1.push(Neighbor{ID: id, Dist: n - 2*d1})
				t2.push(Neighbor{ID: id, Dist: n - 2*d2})
				t3.push(Neighbor{ID: id, Dist: n - 2*d3})
			}
		}
		for ; qi < len(qs); qi++ {
			q, t := qs[qi], tops[qi]
			for i := b; i < end; i++ {
				t.push(Neighbor{ID: int32(i), Dist: e.norms[i] - 2*dot(q, e.m.Row(i))})
			}
		}
	}
}

// dot4 は x を 1 回読むあいだに 4 本のクエリと内積を取る。x の各成分をレジスタに置いたまま 4 回使うので、積和 1 回あたりの読み出しが減る。
func dot4(x, q0, q1, q2, q3 []float32) (s0, s1, s2, s3 float32) {
	n := len(x)
	q0, q1, q2, q3 = q0[:n], q1[:n], q2[:n], q3[:n]
	for j, v := range x {
		s0 += v * q0[j]
		s1 += v * q1[j]
		s2 += v * q2[j]
		s3 += v * q3[j]
	}
	return
}

// SearchBatchParallel は行を CPU の数に分けて SearchBatch と同じ走査をし、各部分の上位 k 件を併合する。同点は ID で決まるので併合の順に依らない。
func (e *Exact) SearchBatchParallel(qs [][]float32, k int) [][]Neighbor {
	for _, q := range qs {
		e.check(q)
	}
	k = min(k, e.m.N)
	var (
		mu    sync.Mutex
		parts [][]*kBest
	)
	parallelRange(e.m.N, func(lo, hi int) {
		tops := make([]*kBest, len(qs))
		for i := range tops {
			tops[i] = newKBest(k)
		}
		e.scanBatch(qs, lo, hi, tops)
		mu.Lock()
		parts = append(parts, tops)
		mu.Unlock()
	})
	out := make([][]Neighbor, len(qs))
	for qi, q := range qs {
		t := newKBest(k)
		for _, p := range parts {
			for _, n := range p[qi].h {
				t.push(n)
			}
		}
		out[qi] = e.finish(q, t.sorted())
	}
	return out
}

// Refine は候補の ID だけ正確な距離を取り直し、近い k 件を返す。量子化や符号で絞った候補の並べ直しに使う。
func (e *Exact) Refine(q []float32, ids []int32, k int) []Neighbor {
	e.check(q)
	t := newKBest(min(k, len(ids)))
	for _, id := range ids {
		t.push(Neighbor{ID: id, Dist: e.norms[id] - 2*dot(q, e.m.Row(int(id)))})
	}
	return e.finish(q, t.sorted())
}
