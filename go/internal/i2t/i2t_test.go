package i2t

import (
	"image"
	"image/color"
	"strings"
	"testing"
)

// 正解データと画像が一致していることを確かめる。
//
// ここが崩れていると、下流の測定はすべて意味を失う。
// 「画像に赤い円がある」という正解に対して実際は青が描かれていれば、
// モデルが正しく答えても幻覚として数える。
// 評価基盤で最初に確かめるべきはこの一致になる。
func TestRenderMatchesGroundTruth(t *testing.T) {
	scene := Scene{
		Width: 480, Height: 480,
		Objects: []Object{
			{Shape: Circle, Color: "赤", Pos: Position{0, 0}},
			{Shape: Square, Color: "青", Pos: Position{2, 2}},
			{Shape: Triangle, Color: "緑", Pos: Position{1, 1}},
		},
	}
	img := scene.Render()

	for _, o := range scene.Objects {
		cx := o.Pos.Col*160 + 80
		cy := o.Pos.Row*160 + 80
		got := img.RGBAAt(cx, cy)
		want := palette[o.Color]

		if got != want {
			t.Errorf("%s の %s: 中心 (%d,%d) の色が %v, 期待 %v。"+
				"描画と正解が食い違っている", o.Pos.Describe(), o.Shape, cx, cy, got, want)
		}
	}

	// 物体を置いていない区画は背景のままであるはず。
	// ここが白でないと、隣の区画にはみ出している。
	for _, p := range []Position{{1, 0}, {2, 0}, {0, 2}} {
		cx, cy := p.Col*160+80, p.Row*160+80
		if got := img.RGBAAt(cx, cy); got != (color.RGBA{255, 255, 255, 255}) {
			t.Errorf("%s には何も置いていないのに色が %v", p.Describe(), got)
		}
	}
}

// 形の違いが画像に現れることを確かめる。
// すべて同じ形で描いていても、中心の色だけ見るテストは通ってしまう。
func TestShapesDiffer(t *testing.T) {
	render := func(sh Shape) *image.RGBA {
		return Scene{Width: 240, Height: 240,
			Objects: []Object{{Shape: sh, Color: "黒", Pos: Position{1, 1}}}}.Render()
	}

	count := func(img *image.RGBA) int {
		n := 0
		b := img.Bounds()
		for y := b.Min.Y; y < b.Max.Y; y++ {
			for x := b.Min.X; x < b.Max.X; x++ {
				if img.RGBAAt(x, y).R < 128 {
					n++
				}
			}
		}
		return n
	}

	sq, ci, tr := count(render(Square)), count(render(Circle)), count(render(Triangle))
	if !(sq > ci && ci > tr) {
		t.Errorf("面積の大小が四角 > 円 > 三角 になっていない: 四角 %d / 円 %d / 三角 %d", sq, ci, tr)
	}
}

func TestPositionDescribe(t *testing.T) {
	tests := []struct {
		pos  Position
		want string
	}{
		{Position{0, 0}, "左上"},
		{Position{2, 2}, "右下"},
		{Position{1, 1}, "中央"},
		{Position{1, 0}, "上中央"},
		{Position{0, 1}, "左中央"},
	}
	for _, tt := range tests {
		if got := tt.pos.Describe(); got != tt.want {
			t.Errorf("%+v = %q, 期待 %q", tt.pos, got, tt.want)
		}
	}
}

func TestMirrorH(t *testing.T) {
	scene := Scene{Width: 240, Height: 240, Objects: []Object{
		{Shape: Circle, Color: "赤", Pos: Position{0, 0}},
	}}
	m := scene.MirrorH()

	if got := m.Objects[0].Pos.Describe(); got != "右上" {
		t.Errorf("左上を反転したら %q, 期待 %q", got, "右上")
	}
	// 2 回反転すれば元に戻る。
	if got := m.MirrorH().Objects[0].Pos; got != scene.Objects[0].Pos {
		t.Errorf("2 回反転して %+v, 期待 %+v", got, scene.Objects[0].Pos)
	}

	// 画像側も同じ位置に来るはず。正解だけ反転して画像がそのままだと
	// すべての測定が「モデルが間違えた」ことになる。
	img := MirrorH(scene.Render())
	cx, cy := m.Objects[0].Pos.Col*80+40, m.Objects[0].Pos.Row*80+40
	if got := img.RGBAAt(cx, cy); got != palette["赤"] {
		t.Errorf("画像の反転が正解の反転と一致しない。%s の色が %v", m.Objects[0].Pos.Describe(), got)
	}
}

func TestExtractMentions(t *testing.T) {
	tests := []struct {
		caption string
		want    []Mention
	}{
		{"赤い円が左上にあります", []Mention{{Shape: Circle, Color: "赤", Raw: "円"}}},
		{"青の四角と緑の三角", []Mention{
			{Shape: Square, Color: "青", Raw: "四角"},
			{Shape: Triangle, Color: "緑", Raw: "三角"},
		}},
		// 色の言及が無い場合。色を勝手に補わない。
		{"円が中央にあります", []Mention{{Shape: Circle, Raw: "円"}}},
		{"何も写っていません", nil},
	}

	for _, tt := range tests {
		got := ExtractMentions(tt.caption)
		if len(got) != len(tt.want) {
			t.Errorf("%q: %d 件, 期待 %d 件 (%+v)", tt.caption, len(got), len(tt.want), got)
			continue
		}
		for i := range got {
			if got[i].Shape != tt.want[i].Shape || got[i].Color != tt.want[i].Color {
				t.Errorf("%q の %d 件目 = %+v, 期待 %+v", tt.caption, i, got[i], tt.want[i])
			}
		}
	}
}

var scene3 = Scene{Objects: []Object{
	{Shape: Circle, Color: "赤", Pos: Position{0, 0}},
	{Shape: Square, Color: "青", Pos: Position{2, 2}},
}}

func TestCHAIR(t *testing.T) {
	tests := []struct {
		name    string
		caption string
		want    float64
		scored  bool
	}{
		{"正しい説明は満点", "赤い円と青い四角があります", 1.0, true},
		{"無い形を挙げれば減点", "赤い円と青い四角と黄の三角があります", 2.0 / 3.0, true},
		{"色の取り違えも減点", "青い円と青い四角があります", 0.5, true},
		{"無い色は減点", "紫の円があります", 0.0, true},
		// 何も言わなければ幻覚は 0 件になる。満点にすると
		// 「黙っていれば最高得点」の実装が最良になる。採点対象外にする。
		{"言及が無ければ採点しない", "画像を確認しました", 0, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := CHAIR(scene3, tt.caption, 0.9)
			if got.Scored != tt.scored {
				t.Fatalf("Scored = %v, 期待 %v (%s)", got.Scored, tt.scored, got)
			}
			if !tt.scored {
				return
			}
			if diff := got.Value - tt.want; diff > 1e-9 || diff < -1e-9 {
				t.Errorf("スコア = %v, 期待 %v\n%s", got.Value, tt.want, got)
			}
		})
	}
}

// CHAIR は取りこぼしを測らない。Coverage と対で見る必要がある。
func TestCoverageCatchesWhatCHAIRMisses(t *testing.T) {
	// 2 つあるうち 1 つしか言っていない。幻覚は無いので CHAIR は満点。
	caption := "赤い円があります"

	if c := CHAIR(scene3, caption, 0.9); c.Value != 1.0 {
		t.Errorf("CHAIR = %v, 期待 1.0（幻覚は無い）", c.Value)
	}
	cov := Coverage(scene3, caption, 0.9)
	if cov.Value != 0.5 {
		t.Errorf("Coverage = %v, 期待 0.5（2 件中 1 件）\n%s", cov.Value, cov)
	}
	if cov.Passed {
		t.Errorf("取りこぼしているのに合格した:\n%s", cov)
	}
}

func TestMirrorConsistency(t *testing.T) {
	tests := []struct {
		name           string
		orig, mirrored string
		want           float64
		scored         bool
	}{
		{"方向が入れ替われば満点", "円は左上にあります", "円は右上にあります", 1.0, true},
		{"入れ替わらなければ 0", "円は左上にあります", "円は左上にあります", 0.0, true},
		{"方向の記述が無ければ採点しない", "円があります", "円があります", 0, false},
		// 左右が同数だと、追随した場合としなかった場合が同じ数になる。
		// 満点が出るが何も測っていないため採点しない。
		{"左右が同数なら採点しない", "左に円、右に四角", "右に円、左に四角", 0, false},
		{"左右が同数なら追随していなくても採点しない", "左に円、右に四角", "左に円、右に四角", 0, false},
		// 指標は方向語の出現回数を数える。物体ごとに位置を書く形にする。
		// 実際のモデル出力も「左上：赤の円 / 左下：緑の三角 / 右中央：青の四角」の形になる。
		{"非対称なら追随を検出できる",
			"左上に円、左下に四角、右中央に三角",
			"右上に円、右下に四角、左中央に三角", 1.0, true},
		// 追随しない場合でも 0 にはならない。左 2 右 1 のまま変わらなければ
		// diff = |2-1| + |1-2| = 2、総数 6 で 0.67 になる。
		// 語数の一致は位置理解の弱い代理指標で、追随した 1.00 と
		// 追随しない 0.67 を閾値で分ける形になる。
		{"非対称で追随しなければ下がる",
			"左上に円、左下に四角、右中央に三角",
			"左上に円、左下に四角、右中央に三角", 2.0 / 3.0, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := MirrorConsistency(tt.orig, tt.mirrored, 0.8)
			if got.Scored != tt.scored {
				t.Fatalf("Scored = %v, 期待 %v (%s)", got.Scored, tt.scored, got)
			}
			if !tt.scored {
				return
			}
			if diff := got.Value - tt.want; diff > 1e-9 || diff < -1e-9 {
				t.Errorf("スコア = %v, 期待 %v\n%s", got.Value, tt.want, got)
			}
		})
	}
}

func TestCERWER(t *testing.T) {
	tests := []struct {
		want, got string
		cer, wer  float64
	}{
		{"HELLO", "HELLO", 0, 0},
		{"HELLO", "HELL0", 0.2, 1.0},
		{"HELLO", "", 1.0, 1.0},
		{"AB CD", "AB CD", 0, 0},
		{"AB CD", "AB XY", 0.4, 0.5},
	}
	for _, tt := range tests {
		if got := CER(tt.want, tt.got); got != tt.cer {
			t.Errorf("CER(%q,%q) = %v, 期待 %v", tt.want, tt.got, got, tt.cer)
		}
		if got := WER(tt.want, tt.got); got != tt.wer {
			t.Errorf("WER(%q,%q) = %v, 期待 %v", tt.want, tt.got, got, tt.wer)
		}
	}
}

func TestOCRAccuracy(t *testing.T) {
	s := Scene{Texts: []TextItem{{Content: "STOP", Pos: Position{1, 1}}}}

	if got := OCRAccuracy(s, "STOP", 0.9); got.Value != 1.0 || !got.Passed {
		t.Errorf("完全一致なのに %s", got)
	}
	if got := OCRAccuracy(s, "ST0P", 0.9); got.Value != 0.75 {
		t.Errorf("1 文字違いのスコア = %v, 期待 0.75", got.Value)
	}
	// CER は 1 を超えうる。スコアは 0 で下限を切る。
	if got := OCRAccuracy(s, strings.Repeat("X", 50), 0.9); got.Value != 0 {
		t.Errorf("大量に誤読したときのスコア = %v, 期待 0", got.Value)
	}
	if got := OCRAccuracy(Scene{}, "何か", 0.9); got.Scored {
		t.Errorf("文字が無い画像を採点した: %s", got)
	}

	// 区切り文字の違いは誤読として数えない。
	// レイアウト保持性は別の観点で、読み取り精度に混ぜるとぶれる。
	multi := Scene{Texts: []TextItem{
		{Content: "STOP", Pos: Position{1, 0}},
		{Content: "EXIT 42", Pos: Position{1, 2}},
	}}
	for _, got := range []string{"STOP EXIT 42", "STOP\nEXIT 42", "STOP  EXIT 42", "STOP\tEXIT 42"} {
		if s := OCRAccuracy(multi, got, 0.9); s.Value != 1.0 {
			t.Errorf("区切り文字だけが違う %q のスコア = %v, 期待 1.0\n%s", got, s.Value, s)
		}
	}
}

// 変換が画像を実際に変えることを確かめる。
// 変換が恒等写像になっていると、ロバストネスの測定が
// 「同じ画像を 2 回投げただけ」になる。
func TestTransformsActuallyChangeImage(t *testing.T) {
	src := Scene{Width: 240, Height: 240, Objects: []Object{
		{Shape: Circle, Color: "赤", Pos: Position{0, 0}},
	}}.Render()

	for _, tr := range DefaultTransforms() {
		dst := tr.Apply(src)
		if imagesEqual(src, dst) {
			t.Errorf("%s が画像を変えていない", tr.Name)
		}
	}
	if imagesEqual(src, MirrorH(src)) {
		t.Error("左右反転が画像を変えていない")
	}
}

func imagesEqual(a, b *image.RGBA) bool {
	if a.Bounds() != b.Bounds() {
		return false
	}
	for i := range a.Pix {
		if a.Pix[i] != b.Pix[i] {
			return false
		}
	}
	return true
}
