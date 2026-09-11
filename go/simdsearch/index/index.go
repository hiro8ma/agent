package index

import (
	"math"
	"math/rand/v2"

	"github.com/hiro8ma/agent/go/simdsearch/vec"
)

type (
	DotFunc     func(a, b []float32) float32
	DotInt8Func func(a, b []int8) int32
	HammingFunc func(a, b []uint64) int
)

// Index は N 本の Dim 次元ベクトルを 1 本の配列に詰めて持つ。
type Index struct {
	N, Dim int
	Data   []float32
}

func New(data []float32, dim int) *Index {
	return &Index{N: len(data) / dim, Dim: dim, Data: data}
}

func (ix *Index) Vec(i int) []float32 {
	return ix.Data[i*ix.Dim : (i+1)*ix.Dim : (i+1)*ix.Dim]
}

// Search は全ベクトルと内積を取って上位 k 件を返す。
func (ix *Index) Search(q []float32, k int, dot DotFunc) []Result {
	t := newTopK(k)
	for id := range ix.N {
		t.push(id, dot(q, ix.Vec(id)))
	}
	return t.results()
}

// SearchBatch は DB ベクトルを 1 本読むたびに全クエリと内積を取る。運ぶ量は同じまま計算が len(qs) 倍になる。
func (ix *Index) SearchBatch(qs [][]float32, k int, dot DotFunc) [][]Result {
	tops := make([]*topK, len(qs))
	for i := range tops {
		tops[i] = newTopK(k)
	}
	for id := range ix.N {
		d := ix.Vec(id)
		for i, q := range qs {
			tops[i].push(id, dot(q, d))
		}
	}
	out := make([][]Result, len(qs))
	for i, t := range tops {
		out[i] = t.results()
	}
	return out
}

// Int8Index はベクトルごとに scale を持つ対称 int8 量子化。運ぶ量は fp32 の 1/4。
type Int8Index struct {
	N, Dim int
	Codes  []int8
	Scales []float32
}

func NewInt8(ix *Index) *Int8Index {
	x := &Int8Index{N: ix.N, Dim: ix.Dim, Codes: make([]int8, len(ix.Data)), Scales: make([]float32, ix.N)}
	for i := range ix.N {
		x.Scales[i] = vec.QuantizeInt8(ix.Vec(i), x.Codes[i*ix.Dim:(i+1)*ix.Dim])
	}
	return x
}

func (x *Int8Index) Search(q []float32, k int, dot DotInt8Func) []Result {
	qc := make([]int8, len(q))
	qs := vec.QuantizeInt8(q, qc)
	t := newTopK(k)
	for id := range x.N {
		c := x.Codes[id*x.Dim : (id+1)*x.Dim : (id+1)*x.Dim]
		t.push(id, float32(dot(qc, c))*qs*x.Scales[id])
	}
	return t.results()
}

// BinaryIndex は符号 1bit の量子化。運ぶ量は fp32 の 1/32 で、元の Index を rerank に使う。
type BinaryIndex struct {
	N, Words int
	Codes    []uint64
	src      *Index
}

func NewBinary(ix *Index) *BinaryIndex {
	w := (ix.Dim + 63) / 64
	x := &BinaryIndex{N: ix.N, Words: w, Codes: make([]uint64, ix.N*w), src: ix}
	for i := range ix.N {
		vec.QuantizeBinary(ix.Vec(i), x.Codes[i*w:(i+1)*w])
	}
	return x
}

// Search はハミング距離の小さい順に返す。スコアは -距離。
func (x *BinaryIndex) Search(q []float32, k int, ham HammingFunc) []Result {
	qc := make([]uint64, x.Words)
	vec.QuantizeBinary(q, qc)
	t := newTopK(k)
	for id := range x.N {
		t.push(id, -float32(ham(qc, x.Codes[id*x.Words:(id+1)*x.Words:(id+1)*x.Words])))
	}
	return t.results()
}

// SearchRerank は 1bit で k*factor 件に絞り、その候補だけ fp32 の内積で採点し直す。
func (x *BinaryIndex) SearchRerank(q []float32, k, factor int, ham HammingFunc, dot DotFunc) []Result {
	t := newTopK(k)
	for _, c := range x.Search(q, k*factor, ham) {
		t.push(c.ID, dot(q, x.src.Vec(c.ID)))
	}
	return t.results()
}

// Clustered は clusters 個の中心の周りに点を散らし、長さ 1 に正規化したデータを作る。
// 一様乱数だと近傍に意味が無く Recall が測れないので、埋め込みに近い「塊のある」分布にする。
func Clustered(n, dim, clusters int, noise float64, seed uint64) []float32 {
	r := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	cents := make([]float64, clusters*dim)
	for i := range cents {
		cents[i] = r.NormFloat64() / math.Sqrt(float64(dim))
	}
	out := make([]float32, n*dim)
	row := make([]float64, dim)
	for i := range n {
		c := cents[r.IntN(clusters)*dim:]
		var norm float64
		for j := range dim {
			row[j] = c[j] + noise*r.NormFloat64()/math.Sqrt(float64(dim))
			norm += row[j] * row[j]
		}
		norm = math.Sqrt(norm)
		for j := range dim {
			out[i*dim+j] = float32(row[j] / norm)
		}
	}
	return out
}
