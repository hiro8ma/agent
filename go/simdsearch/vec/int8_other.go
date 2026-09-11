//go:build !(goexperiment.simd && arm64)

package vec

func DotInt8SIMD(a, b []int8) int32 { return DotInt8Naive(a, b) }
