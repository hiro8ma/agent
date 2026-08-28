// Package i2t は画像からテキストを生成するアプリケーションと、その評価を扱う。
//
// 評価には正解が要る。実写画像を使うと人手でアノテーションを付けることになり、
// アノテーションの品質が評価の上限を決めてしまう。
//
// ここでは画像を合成する。何が・どの色で・どこに描かれているかを
// こちらが決めるため、正解は定義上正しい。ハルシネーション（写っていない物体の記述）や
// 位置の取り違えを、人手を介さず厳密に測れる。
//
// 実写での性能を測れないのが代償になる。合成画像で落ちる実装は
// 実写でも落ちるが、逆は言えない。下限の確認として使う。
package i2t

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"sort"
	"strings"

	"golang.org/x/image/font"
	"golang.org/x/image/font/basicfont"
	"golang.org/x/image/math/fixed"
)

// Shape は描画する形。
type Shape string

const (
	Circle   Shape = "円"
	Square   Shape = "四角"
	Triangle Shape = "三角"
)

// Position は画像内の位置。3x3 に区切って表す。
type Position struct {
	Col int // 0=左 1=中央 2=右
	Row int // 0=上 1=中央 2=下
}

var colNames = [3]string{"左", "中央", "右"}
var rowNames = [3]string{"上", "中央", "下"}

// Describe は位置を日本語で表す。「左上」「中央」「右下」など。
func (p Position) Describe() string {
	if p.Col == 1 && p.Row == 1 {
		return "中央"
	}
	if p.Col == 1 {
		return rowNames[p.Row] + "中央"
	}
	if p.Row == 1 {
		return colNames[p.Col] + "中央"
	}
	return colNames[p.Col] + rowNames[p.Row]
}

// MirrorH は左右反転した位置を返す。鏡像テストで期待値を作るのに使う。
func (p Position) MirrorH() Position {
	return Position{Col: 2 - p.Col, Row: p.Row}
}

// Object は画像に描く 1 つの物体。
type Object struct {
	Shape Shape
	Color string
	Pos   Position
}

// TextItem は画像に描く文字列。OCR 精度の測定に使う。
type TextItem struct {
	Content string
	Pos     Position
}

// Scene は 1 枚の画像とその正解。
type Scene struct {
	Name    string
	Objects []Object
	Texts   []TextItem
	Width   int
	Height  int
}

// 色名と RGBA の対応。説明文に現れる色名と描画色を 1 か所で結ぶ。
var palette = map[string]color.RGBA{
	"赤":    {220, 40, 40, 255},
	"青":    {40, 80, 220, 255},
	"緑":    {40, 170, 70, 255},
	"黄":    {240, 210, 40, 255},
	"黒":    {20, 20, 20, 255},
	"紫":    {150, 60, 190, 255},
	"オレンジ": {240, 140, 40, 255},
}

// ColorNames は使える色名を返す。説明文からの語句抽出に使う。
func ColorNames() []string {
	out := make([]string, 0, len(palette))
	for k := range palette {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// ShapeNames は使える形の名前を返す。
func ShapeNames() []string { return []string{string(Circle), string(Square), string(Triangle)} }

// Contains は指定の形と色の物体が含まれるかを返す。
func (s Scene) Contains(shape Shape, colorName string) bool {
	for _, o := range s.Objects {
		if o.Shape == shape && o.Color == colorName {
			return true
		}
	}
	return false
}

// HasShape は指定の形が含まれるかを返す。
func (s Scene) HasShape(shape Shape) bool {
	for _, o := range s.Objects {
		if o.Shape == shape {
			return true
		}
	}
	return false
}

// HasColor は指定の色が含まれるかを返す。
func (s Scene) HasColor(name string) bool {
	for _, o := range s.Objects {
		if o.Color == name {
			return true
		}
	}
	return false
}

// GroundTruth は正解を人が読める形にする。判定理由の提示に使う。
func (s Scene) GroundTruth() string {
	var b strings.Builder
	for _, o := range s.Objects {
		fmt.Fprintf(&b, "%sの%sが%s / ", o.Color, o.Shape, o.Pos.Describe())
	}
	for _, t := range s.Texts {
		fmt.Fprintf(&b, "文字 %q が%s / ", t.Content, t.Pos.Describe())
	}
	return strings.TrimSuffix(b.String(), " / ")
}

// MirrorH は左右反転した正解を返す。画像側の反転と対で使う。
func (s Scene) MirrorH() Scene {
	out := s
	out.Name = s.Name + "-mirror"
	out.Objects = make([]Object, len(s.Objects))
	for i, o := range s.Objects {
		o.Pos = o.Pos.MirrorH()
		out.Objects[i] = o
	}
	out.Texts = make([]TextItem, len(s.Texts))
	for i, t := range s.Texts {
		t.Pos = t.Pos.MirrorH()
		out.Texts[i] = t
	}
	return out
}

// Render は Scene を画像に描く。
//
// 背景は白。物体は 3x3 の区画の中心に描く。
// 区画を使うのは、位置の記述（左上・中央 など）を正解と機械的に突き合わせるため。
func (s Scene) Render() *image.RGBA {
	w, h := s.Width, s.Height
	if w == 0 {
		w = 480
	}
	if h == 0 {
		h = 480
	}

	img := image.NewRGBA(image.Rect(0, 0, w, h))
	draw.Draw(img, img.Bounds(), &image.Uniform{color.White}, image.Point{}, draw.Src)

	cellW, cellH := w/3, h/3
	size := min(cellW, cellH) / 3

	for _, o := range s.Objects {
		cx := o.Pos.Col*cellW + cellW/2
		cy := o.Pos.Row*cellH + cellH/2
		drawShape(img, o.Shape, palette[o.Color], cx, cy, size)
	}

	for _, t := range s.Texts {
		cx := t.Pos.Col*cellW + cellW/2
		cy := t.Pos.Row*cellH + cellH/2
		drawText(img, t.Content, cx, cy)
	}
	return img
}

func drawShape(img *image.RGBA, shape Shape, c color.RGBA, cx, cy, r int) {
	for y := cy - r; y <= cy+r; y++ {
		for x := cx - r; x <= cx+r; x++ {
			if !image.Pt(x, y).In(img.Bounds()) {
				continue
			}
			dx, dy := x-cx, y-cy
			var inside bool
			switch shape {
			case Circle:
				inside = dx*dx+dy*dy <= r*r
			case Square:
				inside = true
			case Triangle:
				// 頂点が上、底辺が下の二等辺三角形。
				inside = dy >= -r && dy <= r && abs(dx) <= (dy+r)/2
			}
			if inside {
				img.SetRGBA(x, y, c)
			}
		}
	}
}

// drawText はビットマップフォントで文字を描く。
//
// basicfont は ASCII のみ。OCR の測定は英数字に限る。
// 日本語の読み取り精度を測るには別のフォントを埋め込む必要がある。
func drawText(img *image.RGBA, s string, cx, cy int) {
	face := basicfont.Face7x13
	// 拡大して描く。7x13 のままでは小さすぎて読み取りを測れない。
	const scale = 3

	tmp := image.NewRGBA(image.Rect(0, 0, len(s)*7+4, 16))
	draw.Draw(tmp, tmp.Bounds(), &image.Uniform{color.White}, image.Point{}, draw.Src)
	d := &font.Drawer{
		Dst: tmp, Src: &image.Uniform{color.Black}, Face: face,
		Dot: fixed.P(2, 12),
	}
	d.DrawString(s)

	b := tmp.Bounds()
	for y := 0; y < b.Dy(); y++ {
		for x := 0; x < b.Dx(); x++ {
			if tmp.RGBAAt(x, y).R > 128 {
				continue // 白は描かない
			}
			for sy := 0; sy < scale; sy++ {
				for sx := 0; sx < scale; sx++ {
					px := cx - b.Dx()*scale/2 + x*scale + sx
					py := cy - b.Dy()*scale/2 + y*scale + sy
					if image.Pt(px, py).In(img.Bounds()) {
						img.SetRGBA(px, py, color.RGBA{0, 0, 0, 255})
					}
				}
			}
		}
	}
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
