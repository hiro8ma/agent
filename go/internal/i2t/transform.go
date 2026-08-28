package i2t

import (
	"image"
	"image/color"
)

// Transform は画像に加える変換。ロバストネスの測定に使う。
//
// 入力画像が常に理想的な品質で得られるとは限らない。
// 屋外やモバイル端末では、ぼけ・傾き・明るさの変動が入る。
// 変換を加えた画像で説明がどれだけ変わるかを見る。
type Transform struct {
	Name  string
	Apply func(*image.RGBA) *image.RGBA
}

// MirrorH は左右反転する。位置の記述が追随するかを見る。
//
// 他の変換と違い、正解も変わる。Scene.MirrorH と対で使う。
func MirrorH(src *image.RGBA) *image.RGBA {
	b := src.Bounds()
	dst := image.NewRGBA(b)
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			dst.SetRGBA(b.Max.X-1-x, y, src.RGBAAt(x, y))
		}
	}
	return dst
}

// Rotate90 は時計回りに 90 度回す。
func Rotate90(src *image.RGBA) *image.RGBA {
	b := src.Bounds()
	dst := image.NewRGBA(image.Rect(0, 0, b.Dy(), b.Dx()))
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			dst.SetRGBA(b.Max.Y-1-y, x, src.RGBAAt(x, y))
		}
	}
	return dst
}

// Blur は箱ぼかしをかける。radius が大きいほど強い。
func Blur(radius int) func(*image.RGBA) *image.RGBA {
	return func(src *image.RGBA) *image.RGBA {
		b := src.Bounds()
		dst := image.NewRGBA(b)
		for y := b.Min.Y; y < b.Max.Y; y++ {
			for x := b.Min.X; x < b.Max.X; x++ {
				var r, g, bl, n int
				for dy := -radius; dy <= radius; dy++ {
					for dx := -radius; dx <= radius; dx++ {
						p := image.Pt(x+dx, y+dy)
						if !p.In(b) {
							continue
						}
						c := src.RGBAAt(p.X, p.Y)
						r += int(c.R)
						g += int(c.G)
						bl += int(c.B)
						n++
					}
				}
				dst.SetRGBA(x, y, color.RGBA{uint8(r / n), uint8(g / n), uint8(bl / n), 255})
			}
		}
		return dst
	}
}

// Contrast は明暗の差を factor 倍にする。1 未満で薄く、1 より大きいと濃くなる。
func Contrast(factor float64) func(*image.RGBA) *image.RGBA {
	adjust := func(v uint8) uint8 {
		x := (float64(v)-128)*factor + 128
		if x < 0 {
			x = 0
		}
		if x > 255 {
			x = 255
		}
		return uint8(x)
	}
	return func(src *image.RGBA) *image.RGBA {
		b := src.Bounds()
		dst := image.NewRGBA(b)
		for y := b.Min.Y; y < b.Max.Y; y++ {
			for x := b.Min.X; x < b.Max.X; x++ {
				c := src.RGBAAt(x, y)
				dst.SetRGBA(x, y, color.RGBA{adjust(c.R), adjust(c.G), adjust(c.B), 255})
			}
		}
		return dst
	}
}

// Noise は決定的な擬似乱数でノイズを乗せる。
//
// 乱数を使うと実行ごとに画像が変わり、スコアの差が変換由来か
// ノイズ由来か分からなくなる。座標から決まる値を使う。
func Noise(amplitude int) func(*image.RGBA) *image.RGBA {
	return func(src *image.RGBA) *image.RGBA {
		b := src.Bounds()
		dst := image.NewRGBA(b)
		for y := b.Min.Y; y < b.Max.Y; y++ {
			for x := b.Min.X; x < b.Max.X; x++ {
				st := uint64(x)*0x9E3779B97F4A7C15 + uint64(y)*0x2545F4914F6CDD1D
				st ^= st << 13
				st ^= st >> 7
				n := int(st%uint64(2*amplitude+1)) - amplitude

				c := src.RGBAAt(x, y)
				dst.SetRGBA(x, y, color.RGBA{clamp(int(c.R) + n), clamp(int(c.G) + n), clamp(int(c.B) + n), 255})
			}
		}
		return dst
	}
}

func clamp(v int) uint8 {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v)
}

// DefaultTransforms は測定に使う変換の一式。
func DefaultTransforms() []Transform {
	return []Transform{
		{Name: "ぼかし r=2", Apply: Blur(2)},
		{Name: "コントラスト 0.4", Apply: Contrast(0.4)},
		{Name: "ノイズ ±40", Apply: Noise(40)},
		{Name: "90 度回転", Apply: Rotate90},
	}
}
