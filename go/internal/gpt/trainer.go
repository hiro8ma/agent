package gpt

import (
	"fmt"
	"math/rand"
)

// Dataset は自己回帰学習用のトークン列。
// x = tokens[i : i+T]、y = tokens[i+1 : i+T+1] のペアを切り出す（1 トークンずらし）。
type Dataset struct {
	Tokens []int
}

func (d *Dataset) SampleBatch(B, T int, rng *rand.Rand) (x, y []int) {
	if len(d.Tokens) < T+2 {
		panic("dataset too small for block size")
	}
	x = make([]int, B*T)
	y = make([]int, B*T)
	for b := 0; b < B; b++ {
		start := rng.Intn(len(d.Tokens) - T - 1)
		copy(x[b*T:(b+1)*T], d.Tokens[start:start+T])
		copy(y[b*T:(b+1)*T], d.Tokens[start+1:start+T+1])
	}
	return x, y
}

// TrainConfig は学習ループの設定。
type TrainConfig struct {
	BatchSize   int
	Steps       int
	WarmupSteps int
	MaxLR       float64
	MinLR       float64
	WeightDecay float64
	GradClip    float64
	LogEvery    int
	Seed        int64
}

// Train は学習ループを実行し、最終ステップの損失を返す。
// フロー: forward → backward → 勾配クリップ → スケジュール済み LR で AdamW 更新。
func Train(m *Model, ds *Dataset, tc TrainConfig, logf func(format string, args ...any)) float64 {
	if logf == nil {
		logf = func(string, ...any) {}
	}
	rng := rand.New(rand.NewSource(tc.Seed))
	opt := NewAdamW(tc.MaxLR, tc.WeightDecay)
	params := m.Params()
	T := m.Cfg.BlockSize

	lastLoss := 0.0
	for step := 0; step < tc.Steps; step++ {
		x, y := ds.SampleBatch(tc.BatchSize, T, rng)
		m.ZeroGrads()
		loss := m.Forward(x, y, tc.BatchSize, T)
		m.Backward()
		norm := ClipGradNorm(params, tc.GradClip)
		lr := CosineLR(step, tc.WarmupSteps, tc.Steps, tc.MaxLR, tc.MinLR)
		opt.Update(params, lr)
		lastLoss = loss

		if tc.LogEvery > 0 && (step%tc.LogEvery == 0 || step == tc.Steps-1) {
			logf("step %4d | loss %.4f | lr %.5f | grad_norm %.3f", step, loss, lr, norm)
		}
	}
	return lastLoss
}

// NumParams はパラメータ総数を返す（モデル規模の確認用）。
func (m *Model) NumParams() int {
	n := 0
	for _, p := range m.Params() {
		n += len(p.Data)
	}
	return n
}

// String は設定の要約。
func (c Config) String() string {
	return fmt.Sprintf("V=%d block=%d embd=%d layers=%d heads=%d",
		c.VocabSize, c.BlockSize, c.EmbedDim, c.NumLayers, c.NumHeads)
}
