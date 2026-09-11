//go:build goexperiment.simd && arm64

package vec

import (
	"simd/archsimd"
	"testing"
)

// ルーフラインの天井を測る。演算側はレジスタ上の FMA だけ、帯域側は L2 と SLC を溢れる配列の流し読みだけを回す。

const peakInner = 1 << 16

var sinkV archsimd.Float32x4

// bc は初期値を返す。アキュムレータを同じ値で始めると、同じ計算の列がコンパイラに 1 本へまとめられて FMA が消える。
func bc(i int) archsimd.Float32x4 { return archsimd.BroadcastFloat32x4(float32(i) + 0.5) }

//go:noinline
func peakFMA4(iters int) archsimd.Float32x4 {
	m, c := archsimd.BroadcastFloat32x4(0.9999), archsimd.BroadcastFloat32x4(1)
	a0, a1, a2, a3 := bc(0), bc(1), bc(2), bc(3)
	for range iters {
		a0 = a0.MulAdd(m, c)
		a1 = a1.MulAdd(m, c)
		a2 = a2.MulAdd(m, c)
		a3 = a3.MulAdd(m, c)
	}
	return a0.Add(a1).Add(a2).Add(a3)
}

//go:noinline
func peakFMA8(iters int) archsimd.Float32x4 {
	m, c := archsimd.BroadcastFloat32x4(0.9999), archsimd.BroadcastFloat32x4(1)
	a0, a1, a2, a3, a4, a5, a6, a7 := bc(0), bc(1), bc(2), bc(3), bc(4), bc(5), bc(6), bc(7)
	for range iters {
		a0 = a0.MulAdd(m, c)
		a1 = a1.MulAdd(m, c)
		a2 = a2.MulAdd(m, c)
		a3 = a3.MulAdd(m, c)
		a4 = a4.MulAdd(m, c)
		a5 = a5.MulAdd(m, c)
		a6 = a6.MulAdd(m, c)
		a7 = a7.MulAdd(m, c)
	}
	return a0.Add(a1).Add(a2).Add(a3).Add(a4).Add(a5).Add(a6).Add(a7)
}

//go:noinline
func peakFMA16(iters int) archsimd.Float32x4 {
	m, c := archsimd.BroadcastFloat32x4(0.9999), archsimd.BroadcastFloat32x4(1)
	a0, a1, a2, a3, a4, a5, a6, a7 := bc(0), bc(1), bc(2), bc(3), bc(4), bc(5), bc(6), bc(7)
	a8, a9, a10, a11, a12, a13, a14, a15 := bc(8), bc(9), bc(10), bc(11), bc(12), bc(13), bc(14), bc(15)
	for range iters {
		a0 = a0.MulAdd(m, c)
		a1 = a1.MulAdd(m, c)
		a2 = a2.MulAdd(m, c)
		a3 = a3.MulAdd(m, c)
		a4 = a4.MulAdd(m, c)
		a5 = a5.MulAdd(m, c)
		a6 = a6.MulAdd(m, c)
		a7 = a7.MulAdd(m, c)
		a8 = a8.MulAdd(m, c)
		a9 = a9.MulAdd(m, c)
		a10 = a10.MulAdd(m, c)
		a11 = a11.MulAdd(m, c)
		a12 = a12.MulAdd(m, c)
		a13 = a13.MulAdd(m, c)
		a14 = a14.MulAdd(m, c)
		a15 = a15.MulAdd(m, c)
	}
	return a0.Add(a1).Add(a2).Add(a3).Add(a4).Add(a5).Add(a6).Add(a7).
		Add(a8).Add(a9).Add(a10).Add(a11).Add(a12).Add(a13).Add(a14).Add(a15)
}

//go:noinline
func peakFMA24(iters int) archsimd.Float32x4 {
	m, c := archsimd.BroadcastFloat32x4(0.9999), archsimd.BroadcastFloat32x4(1)
	a0, a1, a2, a3, a4, a5, a6, a7 := bc(0), bc(1), bc(2), bc(3), bc(4), bc(5), bc(6), bc(7)
	a8, a9, a10, a11, a12, a13, a14, a15 := bc(8), bc(9), bc(10), bc(11), bc(12), bc(13), bc(14), bc(15)
	a16, a17, a18, a19, a20, a21, a22, a23 := bc(16), bc(17), bc(18), bc(19), bc(20), bc(21), bc(22), bc(23)
	for range iters {
		a0 = a0.MulAdd(m, c)
		a1 = a1.MulAdd(m, c)
		a2 = a2.MulAdd(m, c)
		a3 = a3.MulAdd(m, c)
		a4 = a4.MulAdd(m, c)
		a5 = a5.MulAdd(m, c)
		a6 = a6.MulAdd(m, c)
		a7 = a7.MulAdd(m, c)
		a8 = a8.MulAdd(m, c)
		a9 = a9.MulAdd(m, c)
		a10 = a10.MulAdd(m, c)
		a11 = a11.MulAdd(m, c)
		a12 = a12.MulAdd(m, c)
		a13 = a13.MulAdd(m, c)
		a14 = a14.MulAdd(m, c)
		a15 = a15.MulAdd(m, c)
		a16 = a16.MulAdd(m, c)
		a17 = a17.MulAdd(m, c)
		a18 = a18.MulAdd(m, c)
		a19 = a19.MulAdd(m, c)
		a20 = a20.MulAdd(m, c)
		a21 = a21.MulAdd(m, c)
		a22 = a22.MulAdd(m, c)
		a23 = a23.MulAdd(m, c)
	}
	return a0.Add(a1).Add(a2).Add(a3).Add(a4).Add(a5).Add(a6).Add(a7).
		Add(a8).Add(a9).Add(a10).Add(a11).Add(a12).Add(a13).Add(a14).Add(a15).
		Add(a16).Add(a17).Add(a18).Add(a19).Add(a20).Add(a21).Add(a22).Add(a23)
}

func benchPeak(b *testing.B, accs int, f func(int) archsimd.Float32x4) {
	for b.Loop() {
		sinkV = f(peakInner)
	}
	flop := float64(b.N) * peakInner * float64(accs) * 4 * 2
	b.ReportMetric(flop/b.Elapsed().Seconds()/1e9, "GFLOP/s")
}

func BenchmarkPeakFLOP_4acc(b *testing.B)  { benchPeak(b, 4, peakFMA4) }
func BenchmarkPeakFLOP_8acc(b *testing.B)  { benchPeak(b, 8, peakFMA8) }
func BenchmarkPeakFLOP_16acc(b *testing.B) { benchPeak(b, 16, peakFMA16) }
func BenchmarkPeakFLOP_24acc(b *testing.B) { benchPeak(b, 24, peakFMA24) }

// memN は 256MB。M5 Pro の L2（16MB）と SLC を確実に溢れさせて DRAM を測る。
const memN = 1 << 26

//go:noinline
func readSum(x []float32) archsimd.Float32x4 {
	var a0, a1, a2, a3, a4, a5, a6, a7 archsimd.Float32x4
	for len(x) >= 32 {
		a0 = archsimd.LoadFloat32x4(x).Add(a0)
		a1 = archsimd.LoadFloat32x4(x[4:]).Add(a1)
		a2 = archsimd.LoadFloat32x4(x[8:]).Add(a2)
		a3 = archsimd.LoadFloat32x4(x[12:]).Add(a3)
		a4 = archsimd.LoadFloat32x4(x[16:]).Add(a4)
		a5 = archsimd.LoadFloat32x4(x[20:]).Add(a5)
		a6 = archsimd.LoadFloat32x4(x[24:]).Add(a6)
		a7 = archsimd.LoadFloat32x4(x[28:]).Add(a7)
		x = x[32:]
	}
	return a0.Add(a1).Add(a2).Add(a3).Add(a4).Add(a5).Add(a6).Add(a7)
}

func BenchmarkPeakReadBW(b *testing.B) {
	x := make([]float32, memN)
	for i := range x {
		x[i] = 1
	}
	for b.Loop() {
		sinkV = readSum(x)
	}
	b.ReportMetric(float64(b.N)*memN*4/b.Elapsed().Seconds()/1e9, "read-GB/s")
}
