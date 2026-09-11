//go:build !(goexperiment.simd && arm64)

package vec

func HammingSIMD(a, b []uint64) int { return Hamming(a, b) }
