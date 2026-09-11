//go:build goexperiment.simd && arm64

package vec

import (
	"simd/archsimd"
	"testing"
)

// arm64 の FMLA は加算側のレジスタに上書きする。a = a*m + c だと定数 c を毎回コピーする鎖ができるので、
// アキュムレータを加算側に置いた a = m*c + a の形でも測り、FMA そのものの天井と比べる。

//go:noinline
func peakInPlace4(iters int) archsimd.Float32x4 {
	m, c := archsimd.BroadcastFloat32x4(0.9999), archsimd.BroadcastFloat32x4(1e-7)
	a0, a1, a2, a3 := bc(0), bc(1), bc(2), bc(3)
	for range iters {
		a0 = m.MulAdd(c, a0)
		a1 = m.MulAdd(c, a1)
		a2 = m.MulAdd(c, a2)
		a3 = m.MulAdd(c, a3)
	}
	return a0.Add(a1).Add(a2).Add(a3)
}

//go:noinline
func peakInPlace8(iters int) archsimd.Float32x4 {
	m, c := archsimd.BroadcastFloat32x4(0.9999), archsimd.BroadcastFloat32x4(1e-7)
	a0, a1, a2, a3, a4, a5, a6, a7 := bc(0), bc(1), bc(2), bc(3), bc(4), bc(5), bc(6), bc(7)
	for range iters {
		a0 = m.MulAdd(c, a0)
		a1 = m.MulAdd(c, a1)
		a2 = m.MulAdd(c, a2)
		a3 = m.MulAdd(c, a3)
		a4 = m.MulAdd(c, a4)
		a5 = m.MulAdd(c, a5)
		a6 = m.MulAdd(c, a6)
		a7 = m.MulAdd(c, a7)
	}
	return a0.Add(a1).Add(a2).Add(a3).Add(a4).Add(a5).Add(a6).Add(a7)
}

//go:noinline
func peakInPlace16(iters int) archsimd.Float32x4 {
	m, c := archsimd.BroadcastFloat32x4(0.9999), archsimd.BroadcastFloat32x4(1e-7)
	a0, a1, a2, a3, a4, a5, a6, a7 := bc(0), bc(1), bc(2), bc(3), bc(4), bc(5), bc(6), bc(7)
	a8, a9, a10, a11, a12, a13, a14, a15 := bc(8), bc(9), bc(10), bc(11), bc(12), bc(13), bc(14), bc(15)
	for range iters {
		a0 = m.MulAdd(c, a0)
		a1 = m.MulAdd(c, a1)
		a2 = m.MulAdd(c, a2)
		a3 = m.MulAdd(c, a3)
		a4 = m.MulAdd(c, a4)
		a5 = m.MulAdd(c, a5)
		a6 = m.MulAdd(c, a6)
		a7 = m.MulAdd(c, a7)
		a8 = m.MulAdd(c, a8)
		a9 = m.MulAdd(c, a9)
		a10 = m.MulAdd(c, a10)
		a11 = m.MulAdd(c, a11)
		a12 = m.MulAdd(c, a12)
		a13 = m.MulAdd(c, a13)
		a14 = m.MulAdd(c, a14)
		a15 = m.MulAdd(c, a15)
	}
	return a0.Add(a1).Add(a2).Add(a3).Add(a4).Add(a5).Add(a6).Add(a7).
		Add(a8).Add(a9).Add(a10).Add(a11).Add(a12).Add(a13).Add(a14).Add(a15)
}

//go:noinline
func peakInPlace24(iters int) archsimd.Float32x4 {
	m, c := archsimd.BroadcastFloat32x4(0.9999), archsimd.BroadcastFloat32x4(1e-7)
	a0, a1, a2, a3, a4, a5, a6, a7 := bc(0), bc(1), bc(2), bc(3), bc(4), bc(5), bc(6), bc(7)
	a8, a9, a10, a11, a12, a13, a14, a15 := bc(8), bc(9), bc(10), bc(11), bc(12), bc(13), bc(14), bc(15)
	a16, a17, a18, a19, a20, a21, a22, a23 := bc(16), bc(17), bc(18), bc(19), bc(20), bc(21), bc(22), bc(23)
	for range iters {
		a0 = m.MulAdd(c, a0)
		a1 = m.MulAdd(c, a1)
		a2 = m.MulAdd(c, a2)
		a3 = m.MulAdd(c, a3)
		a4 = m.MulAdd(c, a4)
		a5 = m.MulAdd(c, a5)
		a6 = m.MulAdd(c, a6)
		a7 = m.MulAdd(c, a7)
		a8 = m.MulAdd(c, a8)
		a9 = m.MulAdd(c, a9)
		a10 = m.MulAdd(c, a10)
		a11 = m.MulAdd(c, a11)
		a12 = m.MulAdd(c, a12)
		a13 = m.MulAdd(c, a13)
		a14 = m.MulAdd(c, a14)
		a15 = m.MulAdd(c, a15)
		a16 = m.MulAdd(c, a16)
		a17 = m.MulAdd(c, a17)
		a18 = m.MulAdd(c, a18)
		a19 = m.MulAdd(c, a19)
		a20 = m.MulAdd(c, a20)
		a21 = m.MulAdd(c, a21)
		a22 = m.MulAdd(c, a22)
		a23 = m.MulAdd(c, a23)
	}
	return a0.Add(a1).Add(a2).Add(a3).Add(a4).Add(a5).Add(a6).Add(a7).
		Add(a8).Add(a9).Add(a10).Add(a11).Add(a12).Add(a13).Add(a14).Add(a15).
		Add(a16).Add(a17).Add(a18).Add(a19).Add(a20).Add(a21).Add(a22).Add(a23)
}

func BenchmarkPeakFLOP_InPlace_4acc(b *testing.B)  { benchPeak(b, 4, peakInPlace4) }
func BenchmarkPeakFLOP_InPlace_8acc(b *testing.B)  { benchPeak(b, 8, peakInPlace8) }
func BenchmarkPeakFLOP_InPlace_16acc(b *testing.B) { benchPeak(b, 16, peakInPlace16) }
func BenchmarkPeakFLOP_InPlace_24acc(b *testing.B) { benchPeak(b, 24, peakInPlace24) }
