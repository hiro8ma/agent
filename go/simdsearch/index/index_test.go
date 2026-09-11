package index

import (
	"math"
	"math/rand/v2"
	"slices"
	"sort"
	"testing"

	"github.com/hiro8ma/agent/go/simdsearch/vec"
)

// dataset は同じ分布から DB とクエリを切り出す。別の seed で作ると中心がずれて近傍に意味が無くなる。
func dataset(n, q, dim int, seed uint64) (*Index, [][]float32) {
	data := Clustered(n+q, dim, max(n/100, 1), 1.0, seed)
	ix := New(data[:n*dim], dim)
	qs := make([][]float32, q)
	for i := range q {
		qs[i] = data[(n+i)*dim : (n+i+1)*dim]
	}
	return ix, qs
}

func TestTopKMatchesSort(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 1))
	scores := make([]Result, 500)
	tk := newTopK(10)
	for i := range scores {
		// 同点を混ぜて、同点時の順序も確かめる。
		scores[i] = Result{ID: i, Score: float32(r.IntN(50))}
		tk.push(i, scores[i].Score)
	}
	sort.Slice(scores, func(a, b int) bool { return worse(scores[b], scores[a]) })
	if got := tk.results(); !slices.Equal(got, scores[:10]) {
		t.Fatalf("got %v\nwant %v", got, scores[:10])
	}
}

func TestSearchReturnsTrueTopK(t *testing.T) {
	ix, qs := dataset(3000, 5, 384, 7)
	const k = 10
	impls := map[string]DotFunc{
		"Naive": vec.DotNaive, "Unroll4": vec.DotUnroll4, "SIMD1": vec.DotSIMD1,
		"SIMD": vec.DotSIMD, "Portable": vec.DotPortable, "SIMD16A": vec.DotSIMD16A,
	}
	for qi, q := range qs {
		exact := make([]float64, ix.N)
		for id := range ix.N {
			for j, x := range ix.Vec(id) {
				exact[id] += float64(q[j]) * float64(x)
			}
		}
		sorted := slices.Clone(exact)
		slices.Sort(sorted)
		kth := sorted[len(sorted)-k]
		for name, dot := range impls {
			got := ix.Search(q, k, dot)
			if len(got) != k {
				t.Fatalf("%s q%d: %d 件", name, qi, len(got))
			}
			for _, r := range got {
				// 足す順番の違いで近い同点が入れ替わるのは許し、本当の上位 k 件の外は許さない。
				if exact[r.ID] < kth-1e-5 {
					t.Errorf("%s q%d: id=%d の正確なスコア %v が k 番目 %v を下回る", name, qi, r.ID, exact[r.ID], kth)
				}
			}
		}
	}
}

func TestBatchMatchesSingle(t *testing.T) {
	ix, qs := dataset(2000, 8, 384, 11)
	batch := ix.SearchBatch(qs, 10, vec.DotSIMD)
	for i, q := range qs {
		if want := ix.Search(q, 10, vec.DotSIMD); !slices.Equal(batch[i], want) {
			t.Fatalf("q%d: batch %v\nsingle %v", i, batch[i], want)
		}
	}
}

func TestQuantizedSIMDMatchesScalar(t *testing.T) {
	ix, qs := dataset(2000, 4, 384, 13)
	i8, bin := NewInt8(ix), NewBinary(ix)
	for i, q := range qs {
		if a, b := i8.Search(q, 10, vec.DotInt8SIMD), i8.Search(q, 10, vec.DotInt8Naive); !slices.Equal(a, b) {
			t.Errorf("int8 q%d: simd %v\nnaive %v", i, a, b)
		}
		if a, b := bin.Search(q, 10, vec.HammingSIMD), bin.Search(q, 10, vec.Hamming); !slices.Equal(a, b) {
			t.Errorf("binary q%d: simd %v\nscalar %v", i, a, b)
		}
	}
}

func TestRecall(t *testing.T) {
	ix, qs := dataset(20000, 100, 384, 17)
	i8, bin := NewInt8(ix), NewBinary(ix)
	const k, factor = 10, 10
	var rI8, rBin, rRerank float64
	for _, q := range qs {
		want := ix.Search(q, k, vec.DotNaive)
		rI8 += Recall(i8.Search(q, k, vec.DotInt8SIMD), want)
		rBin += Recall(bin.Search(q, k, vec.HammingSIMD), want)
		rRerank += Recall(bin.SearchRerank(q, k, factor, vec.HammingSIMD, vec.DotSIMD), want)
	}
	n := float64(len(qs))
	rI8, rBin, rRerank = rI8/n, rBin/n, rRerank/n
	t.Logf("Recall@%d int8=%.3f binary=%.3f binary+rerank(x%d)=%.3f", k, rI8, rBin, factor, rRerank)
	// 実測は int8=0.986 binary=0.353 rerank=0.999。閾値は実測から 0.05 以上の余裕を取る。
	if rI8 < 0.93 {
		t.Errorf("int8 の recall %.3f が 0.93 を下回った", rI8)
	}
	if rRerank < 0.95 {
		t.Errorf("rerank の recall %.3f が 0.95 を下回った", rRerank)
	}
	if rRerank-rBin < 0.3 {
		t.Errorf("rerank(%.3f) が 1bit 単体(%.3f)を 0.3 以上上回っていない", rRerank, rBin)
	}
	if math.IsNaN(rI8 + rBin + rRerank) {
		t.Fatal("recall が NaN")
	}
}
