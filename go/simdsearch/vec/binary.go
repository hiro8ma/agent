package vec

import "math/bits"

// QuantizeBinary は各要素の符号だけを 1bit に残す。out は len(v)/64 を切り上げた長さが要る。
func QuantizeBinary(v []float32, out []uint64) {
	clear(out)
	for i, x := range v {
		if x > 0 {
			out[i/64] |= 1 << (i % 64)
		}
	}
}

func Hamming(a, b []uint64) int {
	b = b[:len(a)]
	d := 0
	for i := range a {
		d += bits.OnesCount64(a[i] ^ b[i])
	}
	return d
}
