//go:build goexperiment.simd && arm64

package vec

import (
	"math/bits"
	"simd/archsimd"
)

// maxByteBlocks は uint8 レーンにためてよい 16byte ブロックの数。1 ブロックで 1 レーン最大 8 増えるので 31 で 248。
const maxByteBlocks = 31

// HammingSIMD は CNT でバイトごとに数え、uint8 のまま足してから uint16 に広げて合計する。
// Uint8x16.ReduceSum は uint8 で返すので、384bit ぶんを直接集計すると 255 を超えて桁あふれする。
func HammingSIMD(a, b []uint64) int {
	b = b[:len(a)]
	total := 0
	for len(a) >= 2 {
		blocks := min(len(a)/2, maxByteBlocks)
		var acc archsimd.Uint8x16
		for i := range blocks {
			x := archsimd.LoadUint64x2(a[2*i:]).Xor(archsimd.LoadUint64x2(b[2*i:]))
			acc = acc.Add(x.ReshapeToUint8s().OnesCount())
		}
		wide := acc.ExtendLo8ToUint16().Add(acc.HiToLo().ExtendLo8ToUint16())
		total += int(wide.ReduceSum())
		a, b = a[2*blocks:], b[2*blocks:]
	}
	for i := range a {
		total += bits.OnesCount64(a[i] ^ b[i])
	}
	return total
}
