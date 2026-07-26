package gpt

import (
	"math"
	"math/rand"
	"strings"
	"testing"
)

// 手書き backward を数値微分と突き合わせる。
func TestGradCheck(t *testing.T) {
	cfg := Config{VocabSize: 12, BlockSize: 8, EmbedDim: 8, NumLayers: 2, NumHeads: 2}
	rng := rand.New(rand.NewSource(42))
	m := NewModel(cfg, rng)

	B, T := 2, 4
	idx := make([]int, B*T)
	targets := make([]int, B*T)
	for i := range idx {
		idx[i] = rng.Intn(cfg.VocabSize)
		targets[i] = rng.Intn(cfg.VocabSize)
	}

	m.ZeroGrads()
	m.Forward(idx, targets, B, T)
	m.Backward()

	// 各テンソルからランダムに数点ずつ選んで有限差分と比較する
	const h = 1e-6
	const relTol = 1e-4
	checked := 0
	for _, p := range m.Params() {
		for k := 0; k < 3; k++ {
			i := rng.Intn(len(p.Data))
			orig := p.Data[i]

			p.Data[i] = orig + h
			lossPlus := m.Forward(idx, targets, B, T)
			p.Data[i] = orig - h
			lossMinus := m.Forward(idx, targets, B, T)
			p.Data[i] = orig

			numeric := (lossPlus - lossMinus) / (2 * h)
			analytic := p.Grad[i]

			denom := math.Max(math.Abs(numeric)+math.Abs(analytic), 1e-8)
			rel := math.Abs(numeric-analytic) / denom
			if rel > relTol && math.Abs(numeric-analytic) > 1e-7 {
				t.Errorf("%s[%d]: analytic=%.8f numeric=%.8f rel=%.2e", p.Name, i, analytic, numeric, rel)
			}
			checked++
		}
	}
	if checked == 0 {
		t.Fatal("no gradients checked")
	}
	t.Logf("gradient check passed for %d sampled entries", checked)
}

// TestLossDecreases は小さなコーパスで学習し、損失が明確に下がることを確認する。
func TestLossDecreases(t *testing.T) {
	text := strings.Repeat("吾輩は猫である。名前はまだ無い。", 40)
	tok := NewCharTokenizer(text)
	ds := &Dataset{Tokens: tok.Encode(text)}

	cfg := Config{VocabSize: tok.VocabSize(), BlockSize: 16, EmbedDim: 32, NumLayers: 2, NumHeads: 4}
	rng := rand.New(rand.NewSource(1))
	m := NewModel(cfg, rng)

	// 初期損失はほぼ一様分布 ≒ ln(V) のはず
	x, y := ds.SampleBatch(4, cfg.BlockSize, rng)
	initial := m.Forward(x, y, 4, cfg.BlockSize)
	expected := math.Log(float64(tok.VocabSize()))
	if math.Abs(initial-expected) > 1.0 {
		t.Errorf("initial loss %.3f too far from ln(V)=%.3f", initial, expected)
	}

	final := Train(m, ds, TrainConfig{
		BatchSize: 4, Steps: 150, WarmupSteps: 10,
		MaxLR: 1e-2, MinLR: 1e-3, WeightDecay: 0.01, GradClip: 1.0, Seed: 7,
	}, nil)

	if final > initial*0.5 {
		t.Errorf("loss did not decrease enough: initial=%.4f final=%.4f", initial, final)
	}
	t.Logf("loss: initial=%.4f final=%.4f", initial, final)
}

// TestGenerate は学習済みモデルの生成が語彙内トークンを返し、シード固定で再現することを確認する。
func TestGenerate(t *testing.T) {
	text := strings.Repeat("abcdefg ", 100)
	tok := NewCharTokenizer(text)
	ds := &Dataset{Tokens: tok.Encode(text)}

	cfg := Config{VocabSize: tok.VocabSize(), BlockSize: 8, EmbedDim: 16, NumLayers: 1, NumHeads: 2}
	m := NewModel(cfg, rand.New(rand.NewSource(3)))
	Train(m, ds, TrainConfig{
		BatchSize: 4, Steps: 80, WarmupSteps: 5,
		MaxLR: 1e-2, MinLR: 1e-3, WeightDecay: 0.01, GradClip: 1.0, Seed: 11,
	}, nil)

	prompt := tok.Encode("abc")
	out1 := m.Generate(prompt, 10, 0.8, 5, rand.New(rand.NewSource(99)))
	out2 := m.Generate(prompt, 10, 0.8, 5, rand.New(rand.NewSource(99)))

	if len(out1) != len(prompt)+10 {
		t.Fatalf("unexpected output length %d", len(out1))
	}
	for i := range out1 {
		if out1[i] != out2[i] {
			t.Fatalf("generation not deterministic with fixed seed at %d", i)
		}
		if out1[i] < 0 || out1[i] >= cfg.VocabSize {
			t.Fatalf("token out of vocab: %d", out1[i])
		}
	}
	t.Logf("generated: %q", tok.Decode(out1))
}
