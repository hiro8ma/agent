package vec

import "math"

// QuantizeInt8 はゼロ点なしの対称量子化。q = round(v/scale)、scale = maxAbs/127 で -127〜127 に収める。
func QuantizeInt8(v []float32, out []int8) (scale float32) {
	var maxAbs float32
	for _, x := range v {
		maxAbs = max(maxAbs, float32(math.Abs(float64(x))))
	}
	if maxAbs == 0 {
		clear(out[:len(v)])
		return 0
	}
	scale = maxAbs / 127
	for i, x := range v {
		out[i] = int8(math.Round(float64(x / scale)))
	}
	return scale
}

func DotInt8Naive(a, b []int8) int32 {
	b = b[:len(a)]
	var sum int32
	for i := range a {
		sum += int32(a[i]) * int32(b[i])
	}
	return sum
}
