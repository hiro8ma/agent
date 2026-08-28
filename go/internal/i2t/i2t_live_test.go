package i2t_test

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/hiro8ma/agent/go/internal/i2t"
)

const model = "gemini-3.6-flash"

// 合成画像に対する I2T の評価を実測する。
//
//	GEMINI_API_KEY=... go test -run Live -v ./internal/i2t/
//
// 無料枠は 1 日 20 リクエスト。下の一式で 8 回使う。
// CI では EVAL_REQUIRE=1 を立てて skip を失敗に変える。skip は緑になるため、
// 「評価に通った」と「評価が走らなかった」がテスト結果の色で区別できない。
func describer(t *testing.T) *i2t.Describer {
	t.Helper()

	key := os.Getenv("GEMINI_API_KEY")
	if key == "" {
		if os.Getenv("EVAL_REQUIRE") == "1" {
			t.Fatal("GEMINI_API_KEY が未設定。EVAL_REQUIRE=1 のため skip せず失敗させた")
		}
		t.Skip("GEMINI_API_KEY が未設定のため skip")
	}

	d, err := i2t.NewDescriber(context.Background(), key, model)
	if err != nil {
		t.Fatalf("生成器の作成に失敗: %v", err)
	}
	// 無料枠は 1 分あたり 5 リクエスト。連続で叩くと 429 になる。
	d.MinInterval = 13 * time.Second
	d.MaxRetries = 3
	return d
}

// 左に 2 つ、右に 1 つ置く。
//
// 左右を同数にすると、鏡像テストで「反転に追随した場合」と「しなかった場合」が
// 同じ言及数になり区別できない。満点が出るが何も測っていない状態になる。
// 実測で左 1 右 1 の配置を使って踏んだ。
var basicScene = i2t.Scene{
	Name: "基本", Width: 480, Height: 480,
	Objects: []i2t.Object{
		{Shape: i2t.Circle, Color: "赤", Pos: i2t.Position{Col: 0, Row: 0}},
		{Shape: i2t.Triangle, Color: "緑", Pos: i2t.Position{Col: 0, Row: 2}},
		{Shape: i2t.Square, Color: "青", Pos: i2t.Position{Col: 2, Row: 1}},
	},
}

// 幻覚と取りこぼしを測る。
func TestLiveCHAIRAndCoverage(t *testing.T) {
	d := describer(t)
	ctx := context.Background()

	caption, dur, err := d.Describe(ctx, basicScene.Render())
	if err != nil {
		t.Fatalf("説明の生成に失敗: %v", err)
	}
	t.Logf("正解: %s", basicScene.GroundTruth())
	t.Logf("説明 (%.1fs): %s", dur.Seconds(), caption)

	chair := i2t.CHAIR(basicScene, caption, 0.9)
	cov := i2t.Coverage(basicScene, caption, 0.9)
	t.Logf("%s", chair)
	t.Logf("%s", cov)

	// CHAIR と Coverage は対で見る。片方だけでは
	// 「黙っていれば幻覚 0」「全部言えば取りこぼし 0」の抜け道が残る。
	if !chair.Passed {
		t.Errorf("画像に無いものを記述している:\n%s", chair)
	}
	if !cov.Passed {
		t.Errorf("画像にあるものを取りこぼしている:\n%s", cov)
	}
}

// 左右反転で方向の記述が入れ替わるかを測る。
//
// 位置の記述が正しいかを直接測るには構文解析が要り、
// 解析器の誤りとモデルの誤りが混ざる。変換に対する性質で見る。
func TestLiveMirrorConsistency(t *testing.T) {
	d := describer(t)
	ctx := context.Background()

	img := basicScene.Render()
	orig, _, err := d.Describe(ctx, img)
	if err != nil {
		t.Fatalf("元画像の説明に失敗: %v", err)
	}
	mirrored, _, err := d.Describe(ctx, i2t.MirrorH(img))
	if err != nil {
		t.Fatalf("反転画像の説明に失敗: %v", err)
	}

	t.Logf("元画像: %s", orig)
	t.Logf("反転後: %s", mirrored)

	s := i2t.MirrorConsistency(orig, mirrored, 0.8)
	t.Logf("%s", s)
	if !s.Scored {
		t.Fatalf("方向の記述が無く採点できない。指示が効いていない可能性がある")
	}
	if !s.Passed {
		t.Errorf("左右反転に方向の記述が追随していない:\n%s", s)
	}

	// 反転後の正解でも幻覚が増えないことを確かめる。
	m := i2t.CHAIR(basicScene.MirrorH(), mirrored, 0.9)
	if !m.Passed {
		t.Errorf("反転画像で幻覚が出ている:\n%s", m)
	}
}

// 画像内の文字の読み取り精度を測る。
func TestLiveOCR(t *testing.T) {
	d := describer(t)

	scene := i2t.Scene{
		Name: "文字", Width: 480, Height: 480,
		Texts: []i2t.TextItem{
			{Content: "STOP", Pos: i2t.Position{Col: 1, Row: 0}},
			{Content: "EXIT 42", Pos: i2t.Position{Col: 1, Row: 2}},
		},
	}

	got, err := d.ExtractText(context.Background(), scene.Render())
	if err != nil {
		t.Fatalf("文字の読み取りに失敗: %v", err)
	}

	s := i2t.OCRAccuracy(scene, got, 0.8)
	t.Logf("%s", s)
	if !s.Passed {
		t.Errorf("読み取り精度が閾値に届かない:\n%s", s)
	}
}

// 画像の劣化に対して説明の質が保たれるかを測る。
//
// 元画像のスコアとの差（Δ）を見る。絶対値ではなく低下率で判断するのは、
// 元画像で取れていない性能は劣化耐性の問題ではないため。
func TestLiveRobustness(t *testing.T) {
	d := describer(t)
	ctx := context.Background()

	img := basicScene.Render()
	base, _, err := d.Describe(ctx, img)
	if err != nil {
		t.Fatalf("元画像の説明に失敗: %v", err)
	}
	baseCov := i2t.Coverage(basicScene, base, 0.9)
	t.Logf("元画像 Coverage %.2f / CHAIR %.2f",
		baseCov.Value, i2t.CHAIR(basicScene, base, 0.9).Value)

	for _, tr := range i2t.DefaultTransforms() {
		// 90 度回転は正解の位置も変わるため、位置を含まない指標だけ見る。
		out, _, err := d.Describe(ctx, tr.Apply(img))
		if err != nil {
			t.Errorf("%s: 説明の生成に失敗: %v", tr.Name, err)
			continue
		}

		cov := i2t.Coverage(basicScene, out, 0.9)
		chair := i2t.CHAIR(basicScene, out, 0.9)
		delta := baseCov.Value - cov.Value
		t.Logf("%-16s Coverage %.2f (Δ %+.2f) / CHAIR %.2f",
			tr.Name, cov.Value, -delta, chair.Value)

		// 劣化そのものは想定内。落ち幅が大きすぎる場合だけ報告する。
		if delta > 0.5 {
			t.Errorf("%s で Coverage が %.2f 低下した。劣化耐性が足りない\n説明: %s",
				tr.Name, delta, out)
		}
	}
	t.Logf("API 呼び出し回数: %d", d.Calls)
}
