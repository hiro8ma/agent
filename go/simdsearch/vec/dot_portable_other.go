//go:build !goexperiment.simd

package vec

var PortableEmulated = true

func DotPortable(a, b []float32) float32 { return DotUnroll4(a, b) }
