package workflow

import "time"

// RetryPolicy はステップ単位のリトライ設定。WHY: 失敗したステップだけを再試行し、
// 全体を最初からやり直さない（Temporal の Activity リトライと同じ粒度）。
type RetryPolicy struct {
	// MaximumAttempts は 1 ステップあたりの最大試行回数（初回含む）。1 以上。
	MaximumAttempts int
	// InitialInterval は初回バックオフ。
	InitialInterval time.Duration
	// BackoffCoefficient は試行ごとの間隔倍率。1.0 で固定間隔。
	BackoffCoefficient float64
	// MaximumInterval はバックオフ上限。0 で無制限。
	MaximumInterval time.Duration
}

// DefaultRetryPolicy は 3 回・指数バックオフのデフォルト。
func DefaultRetryPolicy() RetryPolicy {
	return RetryPolicy{
		MaximumAttempts:    3,
		InitialInterval:    10 * time.Millisecond,
		BackoffCoefficient: 2.0,
		MaximumInterval:    time.Second,
	}
}

// backoff は attempt 回目（1 始まり）失敗後の待機時間を返す。
func (p RetryPolicy) backoff(attempt int) time.Duration {
	if p.InitialInterval <= 0 {
		return 0
	}
	d := float64(p.InitialInterval)
	for i := 1; i < attempt; i++ {
		d *= p.BackoffCoefficient
	}
	out := time.Duration(d)
	if p.MaximumInterval > 0 && out > p.MaximumInterval {
		return p.MaximumInterval
	}
	return out
}

// normalized は不正値を安全側に丸めた RetryPolicy を返す。
func (p RetryPolicy) normalized() RetryPolicy {
	if p.MaximumAttempts < 1 {
		p.MaximumAttempts = 1
	}
	if p.BackoffCoefficient <= 0 {
		p.BackoffCoefficient = 1.0
	}
	return p
}
