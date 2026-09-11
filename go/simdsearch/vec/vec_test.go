package vec

import (
	"math"
	"math/rand/v2"
	"runtime"
	"testing"
)

func randFloats(r *rand.Rand, n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(r.NormFloat64())
	}
	return v
}

func dot64(a, b []float32) float64 {
	var s float64
	for i := range a {
		s += float64(a[i]) * float64(b[i])
	}
	return s
}

func TestBuiltWithSIMD(t *testing.T) {
	if runtime.GOARCH == "arm64" && !HasSIMD {
		t.Fatal("arm64 なのに SIMD の経路が入っていない。GOEXPERIMENT=simd を付けてビルドする")
	}
	t.Logf("HasSIMD=%v PortableEmulated=%v", HasSIMD, PortableEmulated)
}

func TestDotImplementationsAgree(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	impls := map[string]func(a, b []float32) float32{
		"Naive": DotNaive, "Unroll4": DotUnroll4, "SIMD1": DotSIMD1, "SIMD": DotSIMD, "Portable": DotPortable,
		"SIMD4A": DotSIMD4A, "SIMD8A": DotSIMD8A, "SIMD16A": DotSIMD16A,
	}
	for n := range 140 {
		a, b := randFloats(r, n), randFloats(r, n)
		want := dot64(a, b)
		for name, f := range impls {
			got := float64(f(a, b))
			if math.Abs(got-want) > 1e-4*math.Max(1, math.Abs(want)) {
				t.Errorf("%s n=%d: got %v want %v", name, n, got, want)
			}
		}
	}
}

func TestDotInt8MatchesNaive(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 4))
	for n := range 70 {
		a, b := make([]int8, n), make([]int8, n)
		for i := range n {
			a[i], b[i] = int8(r.IntN(256)-128), int8(r.IntN(256)-128)
		}
		if got, want := DotInt8SIMD(a, b), DotInt8Naive(a, b); got != want {
			t.Errorf("n=%d: got %d want %d", n, got, want)
		}
	}
}

func TestQuantizeInt8RoundTrip(t *testing.T) {
	r := rand.New(rand.NewPCG(5, 6))
	v := randFloats(r, 384)
	q := make([]int8, len(v))
	scale := QuantizeInt8(v, q)
	for i := range v {
		if d := math.Abs(float64(v[i] - float32(q[i])*scale)); d > float64(scale)/2+1e-6 {
			t.Fatalf("i=%d: 復元誤差 %v が scale/2=%v を超えた", i, d, scale/2)
		}
	}
	if got := QuantizeInt8(make([]float32, 8), q); got != 0 {
		t.Fatalf("ゼロベクトルの scale は 0 のはず: %v", got)
	}
}

func TestHammingSIMDMatchesScalar(t *testing.T) {
	r := rand.New(rand.NewPCG(7, 8))
	// 桁あふれの境界をまたぐ長さと、全ビットが違う最悪ケースを両方通す。
	for _, n := range []int{0, 1, 2, 3, 6, 7, 61, 62, 63, 64, 65, 200} {
		a, b := make([]uint64, n), make([]uint64, n)
		for i := range n {
			a[i], b[i] = r.Uint64(), r.Uint64()
		}
		if got, want := HammingSIMD(a, b), Hamming(a, b); got != want {
			t.Errorf("random n=%d: got %d want %d", n, got, want)
		}
		for i := range n {
			a[i], b[i] = 0, math.MaxUint64
		}
		if got := HammingSIMD(a, b); got != 64*n {
			t.Errorf("all-diff n=%d: got %d want %d", n, got, 64*n)
		}
	}
}

func TestQuantizeBinary(t *testing.T) {
	v := []float32{1, -1, 0, 2}
	out := []uint64{^uint64(0)}
	QuantizeBinary(v, out)
	if out[0] != 0b1001 {
		t.Fatalf("got %b want 1001（0 は負側に倒し、古いビットは消す）", out[0])
	}
}
