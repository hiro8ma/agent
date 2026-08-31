package evalharness

import (
	"fmt"
	"sort"
	"strings"
	"time"
)

// Quality は評価セット 1 本ぶんの品質指標。
//
// 成功率だけでは運用の判断ができない。
// 同じ成功率でも、反復が 2 回か 10 回か、
// 応答が 1 秒か 30 秒かで採否が変わる。
type Quality struct {
	Cases int
	// Succeeded は成功と判定した件数。
	Succeeded int
	// Blocked はガードレールが止めた件数。成功でも失敗でもない。
	Blocked int

	Iterations []int
	Tokens     []int
	Latencies  []time.Duration

	// Repro は同じ入力を繰り返した一致度。未測定なら Scored が false。
	Repro AgentScore
}

// SuccessRate は成功率。分母から Blocked を除く。
//
// 止めた件数を分母に入れると、ガードレールを厳しくするほど
// 成功率が下がる。安全性と成功率が同じ数字を奪い合う形になる。
func (q Quality) SuccessRate() float64 {
	den := q.Cases - q.Blocked
	if den <= 0 {
		return 0
	}
	return float64(q.Succeeded) / float64(den)
}

// P95Latency は 95 パーセンタイルのレイテンシー。
//
// 平均ではなく分位で見る。エージェントは反復回数で実行時間が跳ねるため、
// 平均は「ほとんどの場合」を表さない。
func (q Quality) P95Latency() time.Duration {
	if len(q.Latencies) == 0 {
		return 0
	}
	s := append([]time.Duration(nil), q.Latencies...)
	sort.Slice(s, func(i, j int) bool { return s[i] < s[j] })
	i := (len(s)*95 + 99) / 100
	if i >= len(s) {
		i = len(s) - 1
	}
	return s[i]
}

// MeanIterations は 1 件あたりの平均反復回数。
func (q Quality) MeanIterations() float64 { return meanInt(q.Iterations) }

// MeanTokens は 1 件あたりの平均トークン数。
func (q Quality) MeanTokens() float64 { return meanInt(q.Tokens) }

func meanInt(v []int) float64 {
	if len(v) == 0 {
		return 0
	}
	sum := 0
	for _, x := range v {
		sum += x
	}
	return float64(sum) / float64(len(v))
}

// Add は 1 件の結果を取り込む。
func (q *Quality) Add(t Trajectory, succeeded, blocked bool) {
	q.Cases++
	if blocked {
		q.Blocked++
	} else if succeeded {
		q.Succeeded++
	}
	q.Iterations = append(q.Iterations, len(t.Steps))

	tokens := 0
	var d time.Duration
	for _, s := range t.Steps {
		tokens += s.Tokens
		d += s.Duration
	}
	q.Tokens = append(q.Tokens, tokens)
	q.Latencies = append(q.Latencies, d)
}

func (q Quality) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, "件数 %d（止めた %d）\n", q.Cases, q.Blocked)
	fmt.Fprintf(&b, "  成功率     %.2f\n", q.SuccessRate())
	fmt.Fprintf(&b, "  反復回数   平均 %.1f\n", q.MeanIterations())
	fmt.Fprintf(&b, "  トークン   平均 %.0f\n", q.MeanTokens())
	fmt.Fprintf(&b, "  レイテンシ p95 %v\n", q.P95Latency().Round(time.Millisecond))
	if q.Repro.Scored {
		fmt.Fprintf(&b, "  再現性     %.2f\n", q.Repro.Value)
	} else {
		b.WriteString("  再現性     未測定\n")
	}
	return b.String()
}
