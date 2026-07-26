package gpt

import "math"

// AdamW。weight decay は Tensor.Decay = true のパラメータのみに適用する。
type AdamW struct {
	LR          float64
	Beta1       float64
	Beta2       float64
	Eps         float64
	WeightDecay float64

	step int
	m    map[*Tensor][]float64
	v    map[*Tensor][]float64
}

func NewAdamW(lr, weightDecay float64) *AdamW {
	return &AdamW{
		LR: lr, Beta1: 0.9, Beta2: 0.95, Eps: 1e-8, WeightDecay: weightDecay,
		m: map[*Tensor][]float64{}, v: map[*Tensor][]float64{},
	}
}

// Update は 1 ステップ分のパラメータ更新を行う。lr はスケジューラ適用後の学習率。
func (o *AdamW) Update(params []*Tensor, lr float64) {
	o.step++
	c1 := 1.0 - math.Pow(o.Beta1, float64(o.step))
	c2 := 1.0 - math.Pow(o.Beta2, float64(o.step))
	for _, p := range params {
		m, ok := o.m[p]
		if !ok {
			m = make([]float64, len(p.Data))
			o.m[p] = m
			o.v[p] = make([]float64, len(p.Data))
		}
		v := o.v[p]
		for i := range p.Data {
			g := p.Grad[i]
			m[i] = o.Beta1*m[i] + (1-o.Beta1)*g
			v[i] = o.Beta2*v[i] + (1-o.Beta2)*g*g
			mHat := m[i] / c1
			vHat := v[i] / c2
			update := mHat / (math.Sqrt(vHat) + o.Eps)
			if p.Decay {
				update += o.WeightDecay * p.Data[i]
			}
			p.Data[i] -= lr * update
		}
	}
}

// ClipGradNorm はグローバル L2 ノルムを maxNorm 以下に抑え、元のノルムを返す。
func ClipGradNorm(params []*Tensor, maxNorm float64) float64 {
	total := 0.0
	for _, p := range params {
		for _, g := range p.Grad {
			total += g * g
		}
	}
	norm := math.Sqrt(total)
	if norm > maxNorm {
		scale := maxNorm / (norm + 1e-12)
		for _, p := range params {
			for i := range p.Grad {
				p.Grad[i] *= scale
			}
		}
	}
	return norm
}

// CosineLR は線形ウォームアップ + コサイン減衰の学習率スケジュール。
func CosineLR(step, warmupSteps, maxSteps int, maxLR, minLR float64) float64 {
	if step < warmupSteps {
		return maxLR * float64(step+1) / float64(warmupSteps)
	}
	if step >= maxSteps {
		return minLR
	}
	ratio := float64(step-warmupSteps) / float64(maxSteps-warmupSteps)
	coeff := 0.5 * (1.0 + math.Cos(math.Pi*ratio))
	return minLR + coeff*(maxLR-minLR)
}
