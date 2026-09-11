//go:build goexperiment.simd && arm64

package vec

import "simd/archsimd"

// HasSIMD は SIMD の経路でビルドされたかを示す。false のままベンチを取るとフォールバックを測ることになる。
const HasSIMD = true

// DotSIMD1 はアキュムレータ 1 本の Neon 版。FMA が前の結果を待つので、幅は 4 倍でも依存連鎖は残る。
func DotSIMD1(a, b []float32) float32 {
	b = b[:len(a)]
	var acc archsimd.Float32x4
	for len(a) >= 4 {
		acc = archsimd.LoadFloat32x4(a).MulAdd(archsimd.LoadFloat32x4(b), acc)
		a, b = a[4:], b[4:]
	}
	return hsum(acc) + DotNaive(a, b)
}

// DotSIMD はアキュムレータ 4 本の Neon 版。
func DotSIMD(a, b []float32) float32 {
	b = b[:len(a)]
	var acc0, acc1, acc2, acc3 archsimd.Float32x4
	for len(a) >= 16 {
		acc0 = archsimd.LoadFloat32x4(a).MulAdd(archsimd.LoadFloat32x4(b), acc0)
		acc1 = archsimd.LoadFloat32x4(a[4:]).MulAdd(archsimd.LoadFloat32x4(b[4:]), acc1)
		acc2 = archsimd.LoadFloat32x4(a[8:]).MulAdd(archsimd.LoadFloat32x4(b[8:]), acc2)
		acc3 = archsimd.LoadFloat32x4(a[12:]).MulAdd(archsimd.LoadFloat32x4(b[12:]), acc3)
		a, b = a[16:], b[16:]
	}
	for len(a) >= 4 {
		acc0 = archsimd.LoadFloat32x4(a).MulAdd(archsimd.LoadFloat32x4(b), acc0)
		a, b = a[4:], b[4:]
	}
	return hsum(acc0.Add(acc1).Add(acc2.Add(acc3))) + DotNaive(a, b)
}

// ld4 は配列ポインタから読む。スライスで a[4:] と書くと、読むたびに境界チェックとアドレス計算が入る。
func ld4(p *[4]float32) archsimd.Float32x4 { return archsimd.LoadFloat32x4Array(p) }

// DotSIMD4A は DotSIMD と同じアキュムレータ 4 本で、1 周の境界チェックを 1 回にした版。
func DotSIMD4A(a, b []float32) float32 {
	b = b[:len(a)]
	var c0, c1, c2, c3 archsimd.Float32x4
	for len(a) >= 16 {
		pa, pb := (*[16]float32)(a), (*[16]float32)(b)
		c0 = ld4((*[4]float32)(pa[0:])).MulAdd(ld4((*[4]float32)(pb[0:])), c0)
		c1 = ld4((*[4]float32)(pa[4:])).MulAdd(ld4((*[4]float32)(pb[4:])), c1)
		c2 = ld4((*[4]float32)(pa[8:])).MulAdd(ld4((*[4]float32)(pb[8:])), c2)
		c3 = ld4((*[4]float32)(pa[12:])).MulAdd(ld4((*[4]float32)(pb[12:])), c3)
		a, b = a[16:], b[16:]
	}
	return hsum(c0.Add(c1).Add(c2.Add(c3))) + DotSIMD1(a, b)
}

// DotSIMD8A はアキュムレータ 8 本。
func DotSIMD8A(a, b []float32) float32 {
	b = b[:len(a)]
	var c0, c1, c2, c3, c4, c5, c6, c7 archsimd.Float32x4
	for len(a) >= 32 {
		pa, pb := (*[32]float32)(a), (*[32]float32)(b)
		c0 = ld4((*[4]float32)(pa[0:])).MulAdd(ld4((*[4]float32)(pb[0:])), c0)
		c1 = ld4((*[4]float32)(pa[4:])).MulAdd(ld4((*[4]float32)(pb[4:])), c1)
		c2 = ld4((*[4]float32)(pa[8:])).MulAdd(ld4((*[4]float32)(pb[8:])), c2)
		c3 = ld4((*[4]float32)(pa[12:])).MulAdd(ld4((*[4]float32)(pb[12:])), c3)
		c4 = ld4((*[4]float32)(pa[16:])).MulAdd(ld4((*[4]float32)(pb[16:])), c4)
		c5 = ld4((*[4]float32)(pa[20:])).MulAdd(ld4((*[4]float32)(pb[20:])), c5)
		c6 = ld4((*[4]float32)(pa[24:])).MulAdd(ld4((*[4]float32)(pb[24:])), c6)
		c7 = ld4((*[4]float32)(pa[28:])).MulAdd(ld4((*[4]float32)(pb[28:])), c7)
		a, b = a[32:], b[32:]
	}
	s := c0.Add(c1).Add(c2.Add(c3)).Add(c4.Add(c5).Add(c6.Add(c7)))
	return hsum(s) + DotSIMD1(a, b)
}

// DotSIMD16A はアキュムレータ 16 本。コンパイラが 32 個のロードを先に並べ、48 個の値が同時に生きてスタックに退避するので 8 本より遅い。
func DotSIMD16A(a, b []float32) float32 {
	b = b[:len(a)]
	var c0, c1, c2, c3, c4, c5, c6, c7 archsimd.Float32x4
	var c8, c9, c10, c11, c12, c13, c14, c15 archsimd.Float32x4
	for len(a) >= 64 {
		pa, pb := (*[64]float32)(a), (*[64]float32)(b)
		c0 = ld4((*[4]float32)(pa[0:])).MulAdd(ld4((*[4]float32)(pb[0:])), c0)
		c1 = ld4((*[4]float32)(pa[4:])).MulAdd(ld4((*[4]float32)(pb[4:])), c1)
		c2 = ld4((*[4]float32)(pa[8:])).MulAdd(ld4((*[4]float32)(pb[8:])), c2)
		c3 = ld4((*[4]float32)(pa[12:])).MulAdd(ld4((*[4]float32)(pb[12:])), c3)
		c4 = ld4((*[4]float32)(pa[16:])).MulAdd(ld4((*[4]float32)(pb[16:])), c4)
		c5 = ld4((*[4]float32)(pa[20:])).MulAdd(ld4((*[4]float32)(pb[20:])), c5)
		c6 = ld4((*[4]float32)(pa[24:])).MulAdd(ld4((*[4]float32)(pb[24:])), c6)
		c7 = ld4((*[4]float32)(pa[28:])).MulAdd(ld4((*[4]float32)(pb[28:])), c7)
		c8 = ld4((*[4]float32)(pa[32:])).MulAdd(ld4((*[4]float32)(pb[32:])), c8)
		c9 = ld4((*[4]float32)(pa[36:])).MulAdd(ld4((*[4]float32)(pb[36:])), c9)
		c10 = ld4((*[4]float32)(pa[40:])).MulAdd(ld4((*[4]float32)(pb[40:])), c10)
		c11 = ld4((*[4]float32)(pa[44:])).MulAdd(ld4((*[4]float32)(pb[44:])), c11)
		c12 = ld4((*[4]float32)(pa[48:])).MulAdd(ld4((*[4]float32)(pb[48:])), c12)
		c13 = ld4((*[4]float32)(pa[52:])).MulAdd(ld4((*[4]float32)(pb[52:])), c13)
		c14 = ld4((*[4]float32)(pa[56:])).MulAdd(ld4((*[4]float32)(pb[56:])), c14)
		c15 = ld4((*[4]float32)(pa[60:])).MulAdd(ld4((*[4]float32)(pb[60:])), c15)
		a, b = a[64:], b[64:]
	}
	s := c0.Add(c1).Add(c2.Add(c3)).Add(c4.Add(c5).Add(c6.Add(c7)))
	s = s.Add(c8.Add(c9).Add(c10.Add(c11)).Add(c12.Add(c13).Add(c14.Add(c15))))
	return hsum(s) + DotSIMD1(a, b)
}

// hsum は 4 レーンの合計。Float32x4 には ReduceSum が無いので FADDP を 2 回かける。
func hsum(v archsimd.Float32x4) float32 {
	v = v.ConcatAddPairs(v)
	v = v.ConcatAddPairs(v)
	return v.GetElem(0)
}
