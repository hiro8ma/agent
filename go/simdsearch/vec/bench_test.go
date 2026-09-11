package vec

import (
	"math/rand/v2"
	"testing"
)

const benchDim = 384

var (
	sinkF float32
	sinkI int32
	sinkH int
)

func benchDot(b *testing.B, f func(a, b []float32) float32) {
	r := rand.New(rand.NewPCG(1, 1))
	x, y := randFloats(r, benchDim), randFloats(r, benchDim)
	for b.Loop() {
		sinkF = f(x, y)
	}
}

func BenchmarkDotNaive(b *testing.B)    { benchDot(b, DotNaive) }
func BenchmarkDotUnroll4(b *testing.B)  { benchDot(b, DotUnroll4) }
func BenchmarkDotSIMD1(b *testing.B)    { benchDot(b, DotSIMD1) }
func BenchmarkDotSIMD(b *testing.B)     { benchDot(b, DotSIMD) }
func BenchmarkDotPortable(b *testing.B) { benchDot(b, DotPortable) }
func BenchmarkDotSIMD4A(b *testing.B)   { benchDot(b, DotSIMD4A) }
func BenchmarkDotSIMD8A(b *testing.B)   { benchDot(b, DotSIMD8A) }
func BenchmarkDotSIMD16A(b *testing.B)  { benchDot(b, DotSIMD16A) }

func benchDotInt8(b *testing.B, f func(a, b []int8) int32) {
	r := rand.New(rand.NewPCG(2, 2))
	x, y := make([]int8, benchDim), make([]int8, benchDim)
	for i := range benchDim {
		x[i], y[i] = int8(r.IntN(255)-127), int8(r.IntN(255)-127)
	}
	for b.Loop() {
		sinkI = f(x, y)
	}
}

func BenchmarkDotInt8Naive(b *testing.B) { benchDotInt8(b, DotInt8Naive) }
func BenchmarkDotInt8SIMD(b *testing.B)  { benchDotInt8(b, DotInt8SIMD) }

func benchHamming(b *testing.B, f func(a, b []uint64) int) {
	r := rand.New(rand.NewPCG(3, 3))
	x, y := make([]uint64, benchDim/64), make([]uint64, benchDim/64)
	for i := range x {
		x[i], y[i] = r.Uint64(), r.Uint64()
	}
	for b.Loop() {
		sinkH = f(x, y)
	}
}

func BenchmarkHamming(b *testing.B)     { benchHamming(b, Hamming) }
func BenchmarkHammingSIMD(b *testing.B) { benchHamming(b, HammingSIMD) }
