// Package vec は内積とハミング距離を、スカラ / archsimd / ポータブル simd の各実装で持つ。
package vec

// DotNaive は 1 要素ずつ足す基準実装。sum が前の周回に依存するので加算のレイテンシで詰まる。
func DotNaive(a, b []float32) float32 {
	b = b[:len(a)]
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

// DotUnroll4 はスカラのままアキュムレータを 4 本にした対照実験。SIMD の効果と依存連鎖の解消を切り分ける。
func DotUnroll4(a, b []float32) float32 {
	b = b[:len(a)]
	var s0, s1, s2, s3 float32
	for len(a) >= 4 {
		s0 += a[0] * b[0]
		s1 += a[1] * b[1]
		s2 += a[2] * b[2]
		s3 += a[3] * b[3]
		a, b = a[4:], b[4:]
	}
	sum := (s0 + s1) + (s2 + s3)
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}
