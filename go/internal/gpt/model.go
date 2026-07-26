package gpt

import (
	"math"
	"math/rand"
)

// Config は GPT のハイパーパラメータ。
type Config struct {
	VocabSize int // V
	BlockSize int // maxT（コンテキスト長）
	EmbedDim  int // C
	NumLayers int // L
	NumHeads  int // NH
}

// Tensor はパラメータと勾配の組。
// Decay: 線形層の 2D 重みのみ true（bias・LayerNorm・埋め込みは weight decay 対象外）。
type Tensor struct {
	Name  string
	Data  []float64
	Grad  []float64
	Decay bool
}

func newTensor(name string, size int, decay bool) *Tensor {
	return &Tensor{Name: name, Data: make([]float64, size), Grad: make([]float64, size), Decay: decay}
}

// layerParams は Transformer ブロック 1 層分のパラメータ。
type layerParams struct {
	Ln1W, Ln1B         *Tensor // Pre-LN（アテンション前）
	QkvW, QkvB         *Tensor // [3C,C], [3C]
	AttProjW, AttProjB *Tensor // [C,C], [C]
	Ln2W, Ln2B         *Tensor // Pre-LN（MLP 前）
	FcW, FcB           *Tensor // [4C,C], [4C]
	FcProjW, FcProjB   *Tensor // [C,4C], [C]
}

// layerActs は 1 層分の活性値キャッシュ（backward で再利用）。
type layerActs struct {
	Ln1, Ln1Mean, Ln1Rstd []float64
	Qkv                   []float64
	AttY, PreAtt, Att     []float64
	AttProj, Residual2    []float64
	Ln2, Ln2Mean, Ln2Rstd []float64
	Fch, FchGelu, FcProj  []float64
	Residual3             []float64
}

// Model は GPT-2 型（デコーダのみ・Pre-LN・学習可能位置埋め込み・weight tying）。
type Model struct {
	Cfg    Config
	Wte    *Tensor // [V,C] トークン埋め込み（LM ヘッドと共有 = weight tying）
	Wpe    *Tensor // [maxT,C] 学習可能な位置埋め込み
	Layers []layerParams
	LnfW   *Tensor
	LnfB   *Tensor

	// 直近 Forward の活性値（Backward で使用）
	b, t     int
	idx      []int
	targets  []int
	encoded  []float64
	acts     []layerActs
	lnf      []float64
	lnfMean  []float64
	lnfRstd  []float64
	logits   []float64
	probs    []float64
	losses   []float64
	dEncoded []float64
}

// NewModel は N(0, 0.02) で初期化する。残差射影のみ 1/sqrt(2L) スケール。
func NewModel(cfg Config, rng *rand.Rand) *Model {
	C := cfg.EmbedDim
	V := cfg.VocabSize
	m := &Model{Cfg: cfg}

	m.Wte = newTensor("wte", V*C, false)
	m.Wpe = newTensor("wpe", cfg.BlockSize*C, false)
	m.LnfW = newTensor("lnf_w", C, false)
	m.LnfB = newTensor("lnf_b", C, false)

	residScale := 1.0 / math.Sqrt(float64(2*cfg.NumLayers))
	initNormal := func(t *Tensor, std float64) {
		for i := range t.Data {
			t.Data[i] = rng.NormFloat64() * std
		}
	}
	initNormal(m.Wte, 0.02)
	initNormal(m.Wpe, 0.02)
	for i := range m.LnfW.Data {
		m.LnfW.Data[i] = 1.0
	}

	m.Layers = make([]layerParams, cfg.NumLayers)
	for l := range m.Layers {
		p := &m.Layers[l]
		p.Ln1W = newTensor("ln1_w", C, false)
		p.Ln1B = newTensor("ln1_b", C, false)
		p.QkvW = newTensor("qkv_w", 3*C*C, true)
		p.QkvB = newTensor("qkv_b", 3*C, false)
		p.AttProjW = newTensor("attproj_w", C*C, true)
		p.AttProjB = newTensor("attproj_b", C, false)
		p.Ln2W = newTensor("ln2_w", C, false)
		p.Ln2B = newTensor("ln2_b", C, false)
		p.FcW = newTensor("fc_w", 4*C*C, true)
		p.FcB = newTensor("fc_b", 4*C, false)
		p.FcProjW = newTensor("fcproj_w", C*4*C, true)
		p.FcProjB = newTensor("fcproj_b", C, false)

		for i := range p.Ln1W.Data {
			p.Ln1W.Data[i] = 1.0
		}
		for i := range p.Ln2W.Data {
			p.Ln2W.Data[i] = 1.0
		}
		initNormal(p.QkvW, 0.02)
		initNormal(p.AttProjW, 0.02*residScale)
		initNormal(p.FcW, 0.02)
		initNormal(p.FcProjW, 0.02*residScale)
	}
	return m
}

// Params は全パラメータテンソルを列挙する（オプティマイザ・勾配クリッピング用）。
func (m *Model) Params() []*Tensor {
	ps := []*Tensor{m.Wte, m.Wpe, m.LnfW, m.LnfB}
	for l := range m.Layers {
		p := &m.Layers[l]
		ps = append(ps,
			p.Ln1W, p.Ln1B, p.QkvW, p.QkvB, p.AttProjW, p.AttProjB,
			p.Ln2W, p.Ln2B, p.FcW, p.FcB, p.FcProjW, p.FcProjB)
	}
	return ps
}

func (m *Model) ZeroGrads() {
	for _, p := range m.Params() {
		for i := range p.Grad {
			p.Grad[i] = 0
		}
	}
}

func (m *Model) allocActs(B, T int) {
	C := m.Cfg.EmbedDim
	V := m.Cfg.VocabSize
	NH := m.Cfg.NumHeads
	m.b, m.t = B, T
	m.encoded = make([]float64, B*T*C)
	m.dEncoded = make([]float64, B*T*C)
	m.acts = make([]layerActs, m.Cfg.NumLayers)
	for l := range m.acts {
		a := &m.acts[l]
		a.Ln1 = make([]float64, B*T*C)
		a.Ln1Mean = make([]float64, B*T)
		a.Ln1Rstd = make([]float64, B*T)
		a.Qkv = make([]float64, B*T*3*C)
		a.AttY = make([]float64, B*T*C)
		a.PreAtt = make([]float64, B*NH*T*T)
		a.Att = make([]float64, B*NH*T*T)
		a.AttProj = make([]float64, B*T*C)
		a.Residual2 = make([]float64, B*T*C)
		a.Ln2 = make([]float64, B*T*C)
		a.Ln2Mean = make([]float64, B*T)
		a.Ln2Rstd = make([]float64, B*T)
		a.Fch = make([]float64, B*T*4*C)
		a.FchGelu = make([]float64, B*T*4*C)
		a.FcProj = make([]float64, B*T*C)
		a.Residual3 = make([]float64, B*T*C)
	}
	m.lnf = make([]float64, B*T*C)
	m.lnfMean = make([]float64, B*T)
	m.lnfRstd = make([]float64, B*T)
	m.logits = make([]float64, B*T*V)
	m.probs = make([]float64, B*T*V)
	m.losses = make([]float64, B*T)
}

// Forward は損失（targets ありのとき）と logits を計算する。
// idx, targets は [B*T] のトークン ID。targets が nil なら生成用の forward のみ。
func (m *Model) Forward(idx []int, targets []int, B, T int) float64 {
	if T > m.Cfg.BlockSize {
		panic("sequence length exceeds block size")
	}
	C := m.Cfg.EmbedDim
	V := m.Cfg.VocabSize
	NH := m.Cfg.NumHeads
	N := B * T

	if m.b != B || m.t != T || m.encoded == nil {
		m.allocActs(B, T)
	}
	m.idx = idx
	m.targets = targets

	encoderForward(m.encoded, idx, m.Wte.Data, m.Wpe.Data, B, T, C)

	x := m.encoded
	for l := range m.Layers {
		p := &m.Layers[l]
		a := &m.acts[l]
		layernormForward(a.Ln1, a.Ln1Mean, a.Ln1Rstd, x, p.Ln1W.Data, p.Ln1B.Data, N, C)
		matmulForward(a.Qkv, a.Ln1, p.QkvW.Data, p.QkvB.Data, N, C, 3*C)
		attentionForward(a.AttY, a.PreAtt, a.Att, a.Qkv, B, T, C, NH)
		matmulForward(a.AttProj, a.AttY, p.AttProjW.Data, p.AttProjB.Data, N, C, C)
		residualForward(a.Residual2, x, a.AttProj)
		layernormForward(a.Ln2, a.Ln2Mean, a.Ln2Rstd, a.Residual2, p.Ln2W.Data, p.Ln2B.Data, N, C)
		matmulForward(a.Fch, a.Ln2, p.FcW.Data, p.FcB.Data, N, C, 4*C)
		geluForward(a.FchGelu, a.Fch)
		matmulForward(a.FcProj, a.FchGelu, p.FcProjW.Data, p.FcProjB.Data, N, 4*C, C)
		residualForward(a.Residual3, a.Residual2, a.FcProj)
		x = a.Residual3
	}

	layernormForward(m.lnf, m.lnfMean, m.lnfRstd, x, m.LnfW.Data, m.LnfB.Data, N, C)
	// LM ヘッドは Wte を共有（weight tying）
	matmulForward(m.logits, m.lnf, m.Wte.Data, nil, N, C, V)

	if targets == nil {
		return 0
	}
	softmaxForward(m.probs, m.logits, N, V)
	crossentropyForward(m.losses, m.probs, targets, N, V)
	sum := 0.0
	for _, l := range m.losses {
		sum += l
	}
	return sum / float64(N)
}

// Backward は直近の Forward に対する全パラメータ勾配を累積する。
func (m *Model) Backward() {
	B, T := m.b, m.t
	C := m.Cfg.EmbedDim
	V := m.Cfg.VocabSize
	NH := m.Cfg.NumHeads
	N := B * T

	dlogits := make([]float64, N*V)
	crossentropySoftmaxBackward(dlogits, m.probs, m.targets, N, V, 1.0/float64(N))

	dlnf := make([]float64, N*C)
	matmulBackward(dlnf, m.Wte.Grad, nil, dlogits, m.lnf, m.Wte.Data, N, C, V)

	lastX := m.encoded
	if len(m.acts) > 0 {
		lastX = m.acts[len(m.acts)-1].Residual3
	}
	dx := make([]float64, N*C)
	layernormBackward(dx, m.LnfW.Grad, m.LnfB.Grad, dlnf, lastX, m.LnfW.Data, m.lnfMean, m.lnfRstd, N, C)

	for l := m.Cfg.NumLayers - 1; l >= 0; l-- {
		p := &m.Layers[l]
		a := &m.acts[l]
		input := m.encoded
		if l > 0 {
			input = m.acts[l-1].Residual3
		}

		dResidual2 := make([]float64, N*C)
		dFcProj := make([]float64, N*C)
		residualBackward(dResidual2, dFcProj, dx)

		dFchGelu := make([]float64, N*4*C)
		matmulBackward(dFchGelu, p.FcProjW.Grad, p.FcProjB.Grad, dFcProj, a.FchGelu, p.FcProjW.Data, N, 4*C, C)
		dFch := make([]float64, N*4*C)
		geluBackward(dFch, a.Fch, dFchGelu)
		dLn2 := make([]float64, N*C)
		matmulBackward(dLn2, p.FcW.Grad, p.FcB.Grad, dFch, a.Ln2, p.FcW.Data, N, C, 4*C)
		layernormBackward(dResidual2, p.Ln2W.Grad, p.Ln2B.Grad, dLn2, a.Residual2, p.Ln2W.Data, a.Ln2Mean, a.Ln2Rstd, N, C)

		dInput := make([]float64, N*C)
		dAttProj := make([]float64, N*C)
		residualBackward(dInput, dAttProj, dResidual2)

		dAttY := make([]float64, N*C)
		matmulBackward(dAttY, p.AttProjW.Grad, p.AttProjB.Grad, dAttProj, a.AttY, p.AttProjW.Data, N, C, C)
		dQkv := make([]float64, N*3*C)
		dPreAtt := make([]float64, B*NH*T*T)
		dAtt := make([]float64, B*NH*T*T)
		attentionBackward(dQkv, dPreAtt, dAtt, dAttY, a.Qkv, a.Att, B, T, C, NH)
		dLn1 := make([]float64, N*C)
		matmulBackward(dLn1, p.QkvW.Grad, p.QkvB.Grad, dQkv, a.Ln1, p.QkvW.Data, N, C, 3*C)
		layernormBackward(dInput, p.Ln1W.Grad, p.Ln1B.Grad, dLn1, input, p.Ln1W.Data, a.Ln1Mean, a.Ln1Rstd, N, C)

		dx = dInput
	}

	encoderBackward(m.Wte.Grad, m.Wpe.Grad, dx, m.idx, B, T, C)
}
