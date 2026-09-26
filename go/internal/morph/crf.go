package morph

import (
	"math"
	"slices"
)

func logSumExp(xs []float64) float64 {
	m := math.Inf(-1)
	for _, x := range xs {
		m = max(m, x)
	}
	if math.IsInf(m, -1) {
		return m
	}
	s := 0.0
	for _, x := range xs {
		s += math.Exp(x - m)
	}
	return m + math.Log(s)
}

// forwardBackward は対数の前向き確率 alpha と後ろ向き確率 beta と log Z を返す。重みは exp(-コスト)。
func forwardBackward(p Problem) (alpha, beta [][]float64, logZ float64) {
	n, k := len(p.Emit), p.states()
	alpha = make([][]float64, n)
	beta = make([][]float64, n)
	for t := range n {
		alpha[t] = make([]float64, k)
		beta[t] = make([]float64, k)
	}
	buf := make([]float64, k)
	for j := range k {
		alpha[0][j] = -p.start(j) - p.Emit[0][j]
	}
	for t := 1; t < n; t++ {
		for j := range k {
			for i := range k {
				buf[i] = alpha[t-1][i] - p.Trans[i][j]
			}
			alpha[t][j] = logSumExp(buf) - p.Emit[t][j]
		}
	}
	for j := range k {
		beta[n-1][j] = -p.end(j)
	}
	for t := n - 2; t >= 0; t-- {
		for i := range k {
			for j := range k {
				buf[j] = -p.Trans[i][j] - p.Emit[t+1][j] + beta[t+1][j]
			}
			beta[t][i] = logSumExp(buf)
		}
	}
	for j := range k {
		buf[j] = alpha[n-1][j] + beta[n-1][j]
	}
	return alpha, beta, logSumExp(buf)
}

// LogPartition は前向きアルゴリズムで log Z(x) を O(n k²) で求める。
func LogPartition(p Problem) float64 {
	if len(p.Emit) == 0 {
		return 0
	}
	_, _, logZ := forwardBackward(p)
	return logZ
}

// Marginals は位置ごとの品詞の周辺確率 node[t][j] と、隣り合う品詞の周辺確率 edge[t][i][j]（t≥1）を返す。
func Marginals(p Problem) (node [][]float64, edge [][][]float64, logZ float64) {
	n, k := len(p.Emit), p.states()
	if n == 0 {
		return nil, nil, 0
	}
	alpha, beta, logZ := forwardBackward(p)
	node = make([][]float64, n)
	edge = make([][][]float64, n)
	for t := range n {
		node[t] = make([]float64, k)
		for j := range k {
			node[t][j] = math.Exp(alpha[t][j] + beta[t][j] - logZ)
		}
		if t == 0 {
			continue
		}
		edge[t] = make([][]float64, k)
		for i := range k {
			edge[t][i] = make([]float64, k)
			for j := range k {
				edge[t][i][j] = math.Exp(alpha[t-1][i] - p.Trans[i][j] - p.Emit[t][j] + beta[t][j] - logZ)
			}
		}
	}
	return node, edge, logZ
}

// Sentence は区切り済みのトークン列と正解の品詞列。
type Sentence struct {
	Tokens []string
	Labels []string
}

// CRF は線形連鎖の CRF。Params は割り当て（語彙×品詞）、遷移（品詞×品詞）、文頭、文末のコストを並べたもの。
type CRF struct {
	Labels   []string
	Tokens   []string
	Params   []float64
	labelIdx map[string]int
	tokenIdx map[string]int
}

// NewCRF は学習データに出る語と品詞から、コストがすべて 0 の CRF を作る。
func NewCRF(data []Sentence) *CRF {
	m := &CRF{labelIdx: map[string]int{}, tokenIdx: map[string]int{}}
	for _, s := range data {
		for i, tok := range s.Tokens {
			if _, ok := m.tokenIdx[tok]; !ok {
				m.tokenIdx[tok] = len(m.Tokens)
				m.Tokens = append(m.Tokens, tok)
			}
			if _, ok := m.labelIdx[s.Labels[i]]; !ok {
				m.labelIdx[s.Labels[i]] = len(m.Labels)
				m.Labels = append(m.Labels, s.Labels[i])
			}
		}
	}
	k := len(m.Labels)
	m.Params = make([]float64, len(m.Tokens)*k+k*k+2*k)
	return m
}

func (m *CRF) emitIdx(tok, label int) int { return tok*len(m.Labels) + label }
func (m *CRF) transIdx(from, to int) int {
	return len(m.Tokens)*len(m.Labels) + from*len(m.Labels) + to
}

func (m *CRF) startIdx(label int) int { return m.transIdx(0, 0) + len(m.Labels)*len(m.Labels) + label }
func (m *CRF) endIdx(label int) int   { return m.startIdx(0) + len(m.Labels) + label }

// EmitCost は割り当てのコスト C(token, label)。学習データにない語は 0。
func (m *CRF) EmitCost(token, label string) float64 {
	t, ok := m.tokenIdx[token]
	if !ok {
		return 0
	}
	return m.Params[m.emitIdx(t, m.labelIdx[label])]
}

// TransCost は遷移のコスト C(from, to)。from に BOS、to に EOS を渡すと文頭と文末のコストを返す。
func (m *CRF) TransCost(from, to string) float64 {
	switch {
	case from == BOS:
		return m.Params[m.startIdx(m.labelIdx[to])]
	case to == EOS:
		return m.Params[m.endIdx(m.labelIdx[from])]
	default:
		return m.Params[m.transIdx(m.labelIdx[from], m.labelIdx[to])]
	}
}

// Problem は tokens に対するビタビと前向きアルゴリズムの入力を作る。
func (m *CRF) Problem(tokens []string) Problem {
	k := len(m.Labels)
	p := Problem{Emit: make([][]float64, len(tokens)), Trans: make([][]float64, k), Start: make([]float64, k), End: make([]float64, k)}
	for t, tok := range tokens {
		p.Emit[t] = make([]float64, k)
		if ti, ok := m.tokenIdx[tok]; ok {
			copy(p.Emit[t], m.Params[m.emitIdx(ti, 0):m.emitIdx(ti, 0)+k])
		}
	}
	for i := range k {
		p.Trans[i] = slices.Clone(m.Params[m.transIdx(i, 0) : m.transIdx(i, 0)+k])
	}
	copy(p.Start, m.Params[m.startIdx(0):m.startIdx(0)+k])
	copy(p.End, m.Params[m.endIdx(0):m.endIdx(0)+k])
	return p
}

// Tag は tokens の品詞列を、コストの合計の argmin（= P(y|x) の argmax）としてビタビで求める。
func (m *CRF) Tag(tokens []string) []string {
	res := Viterbi(m.Problem(tokens))
	out := make([]string, len(res.Path))
	for i, j := range res.Path {
		out[i] = m.Labels[j]
	}
	return out
}

// NLL は負の対数尤度 Σ(cost(y) + log Z(x)) に L2 正則化 l2/2·Σθ² を足した値と、その勾配を返す。
func (m *CRF) NLL(data []Sentence, l2 float64) (float64, []float64) {
	grad := make([]float64, len(m.Params))
	loss := 0.0
	k := len(m.Labels)
	for _, s := range data {
		p := m.Problem(s.Tokens)
		gold := make([]int, len(s.Labels))
		for i, l := range s.Labels {
			gold[i] = m.labelIdx[l]
		}
		node, edge, logZ := Marginals(p)
		loss += PathCost(p, gold) + logZ
		n := len(gold)
		grad[m.startIdx(gold[0])]++
		grad[m.endIdx(gold[n-1])]++
		for t := range n {
			ti, known := m.tokenIdx[s.Tokens[t]]
			if known {
				grad[m.emitIdx(ti, gold[t])]++
			}
			if t > 0 {
				grad[m.transIdx(gold[t-1], gold[t])]++
			}
			for j := range k {
				if known {
					grad[m.emitIdx(ti, j)] -= node[t][j]
				}
				if t == 0 {
					grad[m.startIdx(j)] -= node[t][j]
				}
				if t == n-1 {
					grad[m.endIdx(j)] -= node[t][j]
				}
				if t > 0 {
					for i := range k {
						grad[m.transIdx(i, j)] -= edge[t][i][j]
					}
				}
			}
		}
	}
	for i, w := range m.Params {
		loss += l2 / 2 * w * w
		grad[i] += l2 * w
	}
	return loss, grad
}

// TrainConfig は勾配降下の設定。
type TrainConfig struct {
	Steps int
	LR    float64
	L2    float64
}

// Train は勾配降下で NLL を最小化し、各ステップの損失を返す。
func (m *CRF) Train(data []Sentence, cfg TrainConfig) []float64 {
	losses := make([]float64, 0, cfg.Steps)
	for range cfg.Steps {
		loss, grad := m.NLL(data, cfg.L2)
		losses = append(losses, loss)
		for i, g := range grad {
			m.Params[i] -= cfg.LR * g
		}
	}
	return losses
}
