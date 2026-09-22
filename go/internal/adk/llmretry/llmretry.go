// Package llmretry は ADK のモデルを包み、Gemini API のエラーの種類に応じて再試行する。
//
// Go の genai（v1.66.0）と ADK Go（v2.2.0）の Gemini のモデルは、生成の呼び出しを再試行しない。
package llmretry

import (
	"context"
	"errors"
	"iter"
	"math/rand/v2"
	"net"
	"strings"
	"time"

	"google.golang.org/adk/v2/model"
	"google.golang.org/genai"
)

// Policy は再試行の回数と待ち時間を決める。回数は最初の呼び出しを含む。
type Policy struct {
	// MaxAttempts は 429 / 503 / 504 / 408 と通信のエラーの試行回数。
	MaxAttempts int
	// ServerErrorAttempts は 500 の試行回数。同じ要求で再現する障害もあるので少なくする。
	ServerErrorAttempts int
	// BaseDelay と MaxDelay は指数バックオフの初期値と上限。サーバーの指示（RetryInfo）が MaxDelay を超えたら諦める。
	BaseDelay, MaxDelay time.Duration
	// Sleep と Jitter はテストで差し替える。
	Sleep  func(ctx context.Context, d time.Duration) error
	Jitter func(d time.Duration) time.Duration
	// OnRetry は再試行を決めたときに、待つ前に呼ぶ。メトリクスの記録に使う。
	OnRetry func(ctx context.Context, model string, err error)
}

// DefaultPolicy は対話で使う既定値。
func DefaultPolicy() Policy {
	return Policy{MaxAttempts: 4, ServerErrorAttempts: 2, BaseDelay: time.Second, MaxDelay: 30 * time.Second}
}

// Decision は 1 回の失敗に対する判断。
type Decision struct {
	Retry bool
	Wait  time.Duration
}

// Decide は attempt 回目（1 から）の呼び出しが err で失敗したときに、再試行するかと待ち時間を返す。
func Decide(err error, attempt int, p Policy) Decision {
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return Decision{}
	}
	var apiErr genai.APIError
	if !errors.As(err, &apiErr) {
		if netErr := net.Error(nil); errors.As(err, &netErr) && attempt < p.MaxAttempts {
			return Decision{Retry: true, Wait: p.backoff(attempt)}
		}
		return Decision{}
	}
	switch apiErr.Code {
	case 429:
		// 日次の上限は待っても回復しない。retryDelay は日次の上限でも数十秒を返す。
		if perDayQuota(apiErr.Details) || attempt >= p.MaxAttempts {
			return Decision{}
		}
		if d, ok := retryDelay(apiErr.Details); ok {
			if d > p.MaxDelay {
				return Decision{}
			}
			return Decision{Retry: true, Wait: d}
		}
		return Decision{Retry: true, Wait: p.backoff(attempt)}
	case 408, 503, 504:
		return Decision{Retry: attempt < p.MaxAttempts, Wait: p.backoff(attempt)}
	case 500:
		return Decision{Retry: attempt < p.ServerErrorAttempts, Wait: p.backoff(attempt)}
	default:
		return Decision{}
	}
}

func (p Policy) backoff(attempt int) time.Duration {
	d := min(p.BaseDelay<<(attempt-1), p.MaxDelay)
	if p.Jitter != nil {
		return p.Jitter(d)
	}
	return d/2 + rand.N(d/2+1) //nolint:gosec // 待ち時間のゆらぎで、秘密ではない
}

func (p Policy) sleep(ctx context.Context, d time.Duration) error {
	if p.Sleep != nil {
		return p.Sleep(ctx, d)
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}

func retryDelay(details []map[string]any) (time.Duration, bool) {
	for _, d := range details {
		if d["@type"] != "type.googleapis.com/google.rpc.RetryInfo" {
			continue
		}
		s, _ := d["retryDelay"].(string)
		if v, err := time.ParseDuration(s); err == nil {
			return v, true
		}
	}
	return 0, false
}

func perDayQuota(details []map[string]any) bool {
	for _, d := range details {
		if d["@type"] != "type.googleapis.com/google.rpc.QuotaFailure" {
			continue
		}
		violations, _ := d["violations"].([]any)
		for _, v := range violations {
			m, _ := v.(map[string]any)
			if id, _ := m["quotaId"].(string); strings.Contains(id, "PerDay") {
				return true
			}
		}
	}
	return false
}

// Wrap は inner を包み、失敗した呼び出しを p に従って再試行するモデルを返す。
//
// ストリーミングで一部を返した後の失敗は再試行しない。やり直すと返した分が重複する。
func Wrap(inner model.LLM, p Policy) model.LLM {
	return &retrying{inner: inner, p: p}
}

type retrying struct {
	inner model.LLM
	p     Policy
}

func (r *retrying) Name() string { return r.inner.Name() }

func (r *retrying) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		for attempt := 1; ; attempt++ {
			var failed error
			yielded := false
			for resp, err := range r.inner.GenerateContent(ctx, req, stream) {
				if err != nil {
					failed = err
					break
				}
				yielded = true
				if !yield(resp, nil) {
					return
				}
			}
			if failed == nil {
				return
			}
			d := Decide(failed, attempt, r.p)
			if yielded || !d.Retry {
				yield(nil, failed)
				return
			}
			if r.p.OnRetry != nil {
				r.p.OnRetry(ctx, req.Model, failed)
			}
			if err := r.p.sleep(ctx, d.Wait); err != nil {
				yield(nil, errors.Join(failed, err))
				return
			}
		}
	}
}
