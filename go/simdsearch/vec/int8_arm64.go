//go:build goexperiment.simd && arm64

package vec

import "simd/archsimd"

// DotInt8SIMD は SMULL で int16 に広げて掛け、SXTL で int32 に広げて足す。
// arm64 の archsimd には SDOT に当たるメソッドが無いので、16 要素に 4 回の加算がかかる。
func DotInt8SIMD(a, b []int8) int32 {
	b = b[:len(a)]
	var acc0, acc1 archsimd.Int32x4
	for len(a) >= 16 {
		va, vb := archsimd.LoadInt8x16(a), archsimd.LoadInt8x16(b)
		lo := va.MulWidenLo(vb)
		hi := va.HiToLo().MulWidenLo(vb.HiToLo())
		acc0 = acc0.Add(lo.ExtendLo4ToInt32()).Add(lo.HiToLo().ExtendLo4ToInt32())
		acc1 = acc1.Add(hi.ExtendLo4ToInt32()).Add(hi.HiToLo().ExtendLo4ToInt32())
		a, b = a[16:], b[16:]
	}
	return acc0.Add(acc1).ReduceSum() + DotInt8Naive(a, b)
}
