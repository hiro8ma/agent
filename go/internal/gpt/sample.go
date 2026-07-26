package gpt

import (
	"math/rand"
	"sort"
)

// Generate はプロンプトのトークン列から続きを自己回帰生成する。
// temperature で分布の鋭さを調整し、topK > 0 なら上位 k 候補のみから抽選する。
func (m *Model) Generate(prompt []int, maxNewTokens int, temperature float64, topK int, rng *rand.Rand) []int {
	if temperature <= 0 {
		temperature = 1e-8 // ほぼ greedy
	}
	tokens := append([]int{}, prompt...)
	if len(tokens) == 0 {
		tokens = []int{0}
	}
	V := m.Cfg.VocabSize

	for n := 0; n < maxNewTokens; n++ {
		// コンテキスト長を超えたら末尾 BlockSize 分に切り詰める
		ctx := tokens
		if len(ctx) > m.Cfg.BlockSize {
			ctx = ctx[len(ctx)-m.Cfg.BlockSize:]
		}
		T := len(ctx)
		m.allocActs(1, T) // 生成は B=1・可変長のため毎回確保する
		m.Forward(ctx, nil, 1, T)

		// 最終位置の logits を温度で割って softmax
		logits := m.logits[(T-1)*V : T*V]
		scaled := make([]float64, V)
		for i, l := range logits {
			scaled[i] = l / temperature
		}
		probs := make([]float64, V)
		softmaxForward(probs, scaled, 1, V)

		// top-k フィルタ
		if topK > 0 && topK < V {
			type kv struct {
				i int
				p float64
			}
			all := make([]kv, V)
			for i, p := range probs {
				all[i] = kv{i, p}
			}
			sort.Slice(all, func(a, b int) bool { return all[a].p > all[b].p })
			keep := make(map[int]bool, topK)
			sum := 0.0
			for _, e := range all[:topK] {
				keep[e.i] = true
				sum += e.p
			}
			for i := range probs {
				if keep[i] {
					probs[i] /= sum
				} else {
					probs[i] = 0
				}
			}
		}

		// 累積分布から抽選
		r := rng.Float64()
		cum := 0.0
		next := V - 1
		for i, p := range probs {
			cum += p
			if r < cum {
				next = i
				break
			}
		}
		tokens = append(tokens, next)
	}
	return tokens
}
