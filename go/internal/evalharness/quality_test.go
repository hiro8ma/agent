package evalharness

import (
	"strings"
	"testing"
	"time"
)

func step(tokens int, ms int, calls ...ToolCall) Step {
	return Step{ToolCalls: calls, Tokens: tokens, Duration: time.Duration(ms) * time.Millisecond}
}

// 止めた件は分母から外す。含めるとガードレールを厳しくするほど
// 成功率が下がり、安全性と成功率が同じ数字を奪い合う。
func TestSuccessRateExcludesBlocked(t *testing.T) {
	var q Quality
	q.Add(Trajectory{Steps: []Step{step(100, 10)}}, true, false)
	q.Add(Trajectory{Steps: []Step{step(100, 10)}}, true, false)
	q.Add(Trajectory{Steps: []Step{step(100, 10)}}, false, false)
	q.Add(Trajectory{Steps: []Step{step(0, 1)}}, false, true)

	if got := q.SuccessRate(); got < 0.66 || got > 0.67 {
		t.Errorf("成功率 %.4f。2/3 を期待", got)
	}
	if q.Cases != 4 || q.Blocked != 1 {
		t.Errorf("件数 %d 止めた %d", q.Cases, q.Blocked)
	}
}

// 反復が跳ねる実行があると平均は実態を表さない。
func TestP95NotMean(t *testing.T) {
	var q Quality
	for i := 0; i < 19; i++ {
		q.Add(Trajectory{Steps: []Step{step(100, 100)}}, true, false)
	}
	long := make([]Step, 20)
	for i := range long {
		long[i] = step(100, 500)
	}
	q.Add(Trajectory{Steps: long}, true, false)

	if p95 := q.P95Latency(); p95 < 10*time.Second {
		t.Errorf("p95 が %v。跳ねた実行を拾えていない", p95)
	}
	if mi := q.MeanIterations(); mi > 2 {
		t.Errorf("平均反復 %.1f", mi)
	}
	t.Logf("平均反復 %.1f / p95 %v", q.MeanIterations(), q.P95Latency())
}

func TestQualityStringShowsUnmeasured(t *testing.T) {
	var q Quality
	q.Add(Trajectory{Steps: []Step{step(50, 5)}}, true, false)
	if !strings.Contains(q.String(), "再現性     未測定") {
		t.Errorf("未測定が表示されない:\n%s", q)
	}

	call := ToolCall{Name: "search", Args: map[string]any{"q": "a"}}
	q.Repro = Reproducibility([]Trajectory{
		repTraj("x", call), repTraj("y", call),
	}, 0.8)
	if !strings.Contains(q.String(), "再現性     1.00") {
		t.Errorf("測定値が表示されない:\n%s", q)
	}
	t.Log("\n" + q.String())
}
