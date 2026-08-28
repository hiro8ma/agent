package i2t

import (
	"strings"
	"testing"

	"pgregory.net/rapid"
)

// 例ベースのテストは特定の画像と説明文でしか確かめられない。
// ここでは指標が満たすべき性質を書き、反例を探させる。

func genScene(t *rapid.T) Scene {
	colors := ColorNames()
	shapes := []Shape{Circle, Square, Triangle}

	genObj := rapid.Custom(func(t *rapid.T) Object {
		return Object{
			Shape: rapid.SampledFrom(shapes).Draw(t, "shape"),
			Color: rapid.SampledFrom(colors).Draw(t, "color"),
			Pos: Position{
				Col: rapid.IntRange(0, 2).Draw(t, "col"),
				Row: rapid.IntRange(0, 2).Draw(t, "row"),
			},
		}
	})

	return Scene{
		Width: 240, Height: 240,
		Objects: rapid.SliceOfN(genObj, 0, 5).Draw(t, "objects"),
	}
}

// 正解どおりの説明文を組み立てる。理想の出力にあたる。
func perfectCaption(s Scene) string {
	var b strings.Builder
	for _, o := range s.Objects {
		b.WriteString(o.Color)
		b.WriteString("の")
		b.WriteString(string(o.Shape))
		b.WriteString("が")
		b.WriteString(o.Pos.Describe())
		b.WriteString("にあります。")
	}
	return b.String()
}

// 正解どおりの説明文は、幻覚も取りこぼしも無い。
//
// 満点を取れない指標は、改善の目標にならない。
func TestPropertyPerfectCaptionScoresFull(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		s := genScene(t)
		if len(s.Objects) == 0 {
			t.Skip("物体が無い画像は採点対象外")
		}
		caption := perfectCaption(s)

		if c := CHAIR(s, caption, 0.9); !c.Scored || c.Value != 1 {
			t.Fatalf("正解どおりなのに CHAIR = %v\n正解: %s\n説明: %s", c.Value, s.GroundTruth(), caption)
		}
		if c := Coverage(s, caption, 0.9); !c.Scored || c.Value != 1 {
			t.Fatalf("正解どおりなのに Coverage = %v\n正解: %s\n説明: %s", c.Value, s.GroundTruth(), caption)
		}
	})
}

// どんな説明文でもスコアは 0..1 に収まる。
func TestPropertyScoresStayInRange(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		s := genScene(t)
		caption := rapid.SampledFrom([]string{
			perfectCaption(s), "赤い円と青い四角と緑の三角",
			"何もありません", "", "円円円円", "紫のオレンジの三角",
		}).Draw(t, "caption")

		for _, sc := range []Score{
			CHAIR(s, caption, 0.9),
			Coverage(s, caption, 0.9),
			OCRAccuracy(s, caption, 0.8),
		} {
			if !sc.Scored {
				if sc.Passed {
					t.Fatalf("%s: 採点していないのに合格になっている", sc.Metric)
				}
				continue
			}
			if sc.Value < 0 || sc.Value > 1 {
				t.Fatalf("%s のスコアが範囲外: %v", sc.Metric, sc.Value)
			}
			if sc.Passed != (sc.Value >= sc.Threshold) {
				t.Fatalf("%s: 合否 %v がスコア %.4f と閾値 %.4f に一致しない",
					sc.Metric, sc.Passed, sc.Value, sc.Threshold)
			}
		}
	})
}

// 説明文に無い形を足すと CHAIR は決して上がらない。
func TestPropertyHallucinationNeverImprovesCHAIR(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		s := genScene(t)
		caption := perfectCaption(s)
		before := CHAIR(s, caption, 0.9)
		if !before.Scored {
			t.Skip("採点対象外")
		}

		// 画像に無い形と色の組み合わせを足す。
		var missing string
		for _, sh := range ShapeNames() {
			if !s.HasShape(Shape(sh)) {
				missing = sh
				break
			}
		}
		if missing == "" {
			t.Skip("すべての形が写っている")
		}

		after := CHAIR(s, caption+"それから"+missing+"もあります。", 0.9)
		if after.Value > before.Value {
			t.Fatalf("無い形を足したのに CHAIR が %.4f → %.4f と上がった", before.Value, after.Value)
		}
	})
}

// 2 回反転すれば元の正解に戻る。
func TestPropertyMirrorIsInvolution(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		s := genScene(t)
		back := s.MirrorH().MirrorH()

		if len(back.Objects) != len(s.Objects) {
			t.Fatalf("物体の数が変わった: %d → %d", len(s.Objects), len(back.Objects))
		}
		for i := range s.Objects {
			if back.Objects[i] != s.Objects[i] {
				t.Fatalf("2 回反転して %+v, 期待 %+v", back.Objects[i], s.Objects[i])
			}
		}
	})
}

// 画像の反転と正解の反転が一致する。
//
// どちらか片方だけ反転すると、すべての測定が
// 「モデルが間違えた」ことになる。評価基盤の側の誤りが
// モデルの誤りとして現れる、最も気づきにくい壊れ方になる。
func TestPropertyImageMirrorMatchesSceneMirror(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		s := genScene(t)
		if len(s.Objects) == 0 {
			t.Skip("物体が無い")
		}

		img := MirrorH(s.Render())
		m := s.MirrorH()
		cell := s.Width / 3

		for _, o := range m.Objects {
			cx, cy := o.Pos.Col*cell+cell/2, o.Pos.Row*cell+cell/2
			got := img.RGBAAt(cx, cy)
			// 同じ区画に複数の物体が重なることがあるため、
			// パレットのいずれかの色であれば良しとする。
			if got == palette[o.Color] {
				continue
			}
			overlapped := false
			for _, other := range m.Objects {
				if other.Pos == o.Pos && got == palette[other.Color] {
					overlapped = true
					break
				}
			}
			if !overlapped {
				t.Fatalf("画像の反転と正解の反転が食い違う。%s の色が %v, 期待 %v",
					o.Pos.Describe(), got, palette[o.Color])
			}
		}
	})
}

// CER と WER の基本性質。
func TestPropertyEditDistance(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		a := rapid.StringMatching(`[A-Z0-9 ]{0,12}`).Draw(t, "a")
		b := rapid.StringMatching(`[A-Z0-9 ]{0,12}`).Draw(t, "b")

		// 同じ文字列なら 0。
		if got := CER(a, a); got != 0 {
			t.Fatalf("CER(%q,%q) = %v, 期待 0", a, a, got)
		}
		if got := WER(a, a); got != 0 {
			t.Fatalf("WER(%q,%q) = %v, 期待 0", a, a, got)
		}
		// 負にはならない。
		if CER(a, b) < 0 || WER(a, b) < 0 {
			t.Fatalf("CER/WER が負: %v / %v", CER(a, b), WER(a, b))
		}
		// 違う文字列なら 0 より大きい。
		if a != b && len([]rune(a)) > 0 && CER(a, b) <= 0 {
			t.Fatalf("違う文字列 %q %q なのに CER = 0", a, b)
		}
	})
}

// 変換は画像の大きさを保つ（回転を除く）。
// 大きさが変わると、区画から位置を求める計算がずれる。
func TestPropertyTransformsPreserveBounds(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		s := genScene(t)
		src := s.Render()

		for _, tr := range DefaultTransforms() {
			if tr.Name == "90 度回転" {
				continue // 回転は縦横が入れ替わる
			}
			if got := tr.Apply(src).Bounds(); got != src.Bounds() {
				t.Fatalf("%s が大きさを変えた: %v → %v", tr.Name, src.Bounds(), got)
			}
		}
		if got := MirrorH(src).Bounds(); got != src.Bounds() {
			t.Fatalf("左右反転が大きさを変えた: %v → %v", src.Bounds(), got)
		}
	})
}
