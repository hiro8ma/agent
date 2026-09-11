//go:build !(goexperiment.simd && arm64)

package vec

const HasSIMD = false

func DotSIMD1(a, b []float32) float32 { return DotNaive(a, b) }

func DotSIMD(a, b []float32) float32 { return DotUnroll4(a, b) }

func DotSIMD4A(a, b []float32) float32 { return DotUnroll4(a, b) }

func DotSIMD8A(a, b []float32) float32 { return DotUnroll4(a, b) }

func DotSIMD16A(a, b []float32) float32 { return DotUnroll4(a, b) }
