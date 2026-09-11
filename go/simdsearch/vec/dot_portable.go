//go:build goexperiment.simd

package vec

import "simd"

// PortableEmulated は simd パッケージがハードの命令を使わず Go で模擬しているかを示す。
var PortableEmulated = simd.Emulated()

// DotPortable はレーン数を型に書かない simd.Float32s 版。幅は実行時に決まる（Neon なら 4）。
func DotPortable(a, b []float32) float32 {
	b = b[:len(a)]
	var acc0, acc1, acc2, acc3 simd.Float32s
	n := acc0.Len()
	for len(a) >= 4*n {
		acc0 = simd.LoadFloat32s(a).MulAdd(simd.LoadFloat32s(b), acc0)
		acc1 = simd.LoadFloat32s(a[n:]).MulAdd(simd.LoadFloat32s(b[n:]), acc1)
		acc2 = simd.LoadFloat32s(a[2*n:]).MulAdd(simd.LoadFloat32s(b[2*n:]), acc2)
		acc3 = simd.LoadFloat32s(a[3*n:]).MulAdd(simd.LoadFloat32s(b[3*n:]), acc3)
		a, b = a[4*n:], b[4*n:]
	}
	var buf [16]float32
	acc0.Add(acc1).Add(acc2.Add(acc3)).Store(buf[:n])
	var sum float32
	for _, v := range buf[:n] {
		sum += v
	}
	return sum + DotNaive(a, b)
}
