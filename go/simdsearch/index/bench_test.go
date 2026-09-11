package index

import (
	"fmt"
	"sync"
	"testing"

	"github.com/hiro8ma/agent/go/simdsearch/vec"
)

// 10 万件 x 384 次元の fp32 は 153.6MB で、M5 Pro の L2（16MB）にも SLC にも収まらない。
const (
	benchN   = 100_000
	benchDim = 384
	benchK   = 10
	benchQ   = 64
)

var (
	benchOnce sync.Once
	bIx       *Index
	bQs       [][]float32
	bI8       *Int8Index
	bBin      *BinaryIndex
	sink      []Result
	sinkBatch [][]Result
)

func setup(b *testing.B) {
	b.Helper()
	benchOnce.Do(func() {
		bIx, bQs = dataset(benchN, benchQ, benchDim, 42)
		bI8, bBin = NewInt8(bIx), NewBinary(bIx)
	})
}

// report はベンチの 1 op あたりの量から、ルーフラインに載せる指標を出す。
func report(b *testing.B, queries int, bytes, ops float64, unit string) {
	sec := b.Elapsed().Seconds() / float64(b.N)
	b.ReportMetric(sec*1e3/float64(queries), "ms/query")
	b.ReportMetric(bytes/sec/1e9, "GB/s")
	if ops > 0 {
		b.ReportMetric(ops/sec/1e9, unit)
		b.ReportMetric(ops/bytes, "AI")
	}
}

const fp32Bytes, fp32Flop = float64(benchN * benchDim * 4), float64(benchN * benchDim * 2)

func benchSearch(b *testing.B, dot DotFunc) {
	setup(b)
	q := bQs[0]
	for b.Loop() {
		sink = bIx.Search(q, benchK, dot)
	}
	report(b, 1, fp32Bytes, fp32Flop, "GFLOP/s")
}

func BenchmarkSearchNaive(b *testing.B)    { benchSearch(b, vec.DotNaive) }
func BenchmarkSearchUnroll4(b *testing.B)  { benchSearch(b, vec.DotUnroll4) }
func BenchmarkSearchSIMD1(b *testing.B)    { benchSearch(b, vec.DotSIMD1) }
func BenchmarkSearchSIMD(b *testing.B)     { benchSearch(b, vec.DotSIMD) }
func BenchmarkSearchPortable(b *testing.B) { benchSearch(b, vec.DotPortable) }
func BenchmarkSearchSIMD4A(b *testing.B)   { benchSearch(b, vec.DotSIMD4A) }
func BenchmarkSearchSIMD8A(b *testing.B)   { benchSearch(b, vec.DotSIMD8A) }
func BenchmarkSearchSIMD16A(b *testing.B)  { benchSearch(b, vec.DotSIMD16A) }

func BenchmarkSearchBatchSIMD8A(b *testing.B) {
	benchBatch(b, vec.DotSIMD8A, []int{1, 2, 4, 8, 16, 32, 64})
}

func BenchmarkSearchBatchSIMD16A(b *testing.B) {
	benchBatch(b, vec.DotSIMD16A, []int{1, 2, 4, 8, 16, 32, 64})
}

func benchBatch(b *testing.B, dot DotFunc, sizes []int) {
	for _, n := range sizes {
		b.Run(fmt.Sprintf("B=%d", n), func(b *testing.B) {
			setup(b)
			qs := bQs[:n]
			for b.Loop() {
				sinkBatch = bIx.SearchBatch(qs, benchK, dot)
			}
			report(b, n, fp32Bytes, float64(n)*fp32Flop, "GFLOP/s")
		})
	}
}

func BenchmarkSearchBatchSIMD(b *testing.B) {
	benchBatch(b, vec.DotSIMD, []int{1, 2, 4, 8, 16, 32, 64})
}
func BenchmarkSearchBatchNaive(b *testing.B) { benchBatch(b, vec.DotNaive, []int{1, 32}) }

func benchInt8(b *testing.B, dot DotInt8Func) {
	setup(b)
	q := bQs[0]
	for b.Loop() {
		sink = bI8.Search(q, benchK, dot)
	}
	report(b, 1, float64(benchN*benchDim+benchN*4), fp32Flop, "Gop/s")
}

func BenchmarkSearchInt8Naive(b *testing.B) { benchInt8(b, vec.DotInt8Naive) }
func BenchmarkSearchInt8SIMD(b *testing.B)  { benchInt8(b, vec.DotInt8SIMD) }

func benchBinary(b *testing.B, ham HammingFunc) {
	setup(b)
	q := bQs[0]
	for b.Loop() {
		sink = bBin.Search(q, benchK, ham)
	}
	report(b, 1, float64(benchN*bBin.Words*8), 0, "")
}

func BenchmarkSearchBinary(b *testing.B)     { benchBinary(b, vec.Hamming) }
func BenchmarkSearchBinarySIMD(b *testing.B) { benchBinary(b, vec.HammingSIMD) }

func BenchmarkSearchBinaryRerank(b *testing.B) {
	setup(b)
	q := bQs[0]
	for b.Loop() {
		sink = bBin.SearchRerank(q, benchK, 10, vec.Hamming, vec.DotSIMD)
	}
	report(b, 1, float64(benchN*bBin.Words*8), 0, "")
}
