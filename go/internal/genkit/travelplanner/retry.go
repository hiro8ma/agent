package travelplanner

import (
	"context"
	"strings"
	"time"
)

type retryPolicy struct {
	maxRetries int
	base       time.Duration
	limit      time.Duration
}

var defaultRetry = retryPolicy{maxRetries: 2, base: 2 * time.Second, limit: 30 * time.Second}

// withRetry は RESOURCE_EXHAUSTED のときだけ呼び直す。
// 生成中の文章を送り始めた後に呼び直すと先頭から送り直して重複するので、streamed が true なら呼び直さない。
func withRetry[T any](ctx context.Context, p retryPolicy, streamed func() bool, call func() (T, error)) (T, error) {
	for attempt := 0; ; attempt++ {
		v, err := call()
		if err == nil {
			return v, nil
		}
		if attempt >= p.maxRetries || streamed() || !isResourceExhausted(err) {
			return v, err
		}
		select {
		case <-ctx.Done():
			return v, ctx.Err()
		case <-time.After(p.wait(attempt)):
		}
	}
}

func (p retryPolicy) wait(attempt int) time.Duration {
	return min(p.base<<attempt, p.limit)
}

func isResourceExhausted(err error) bool {
	msg := err.Error()
	return strings.Contains(msg, "RESOURCE_EXHAUSTED") || strings.Contains(msg, "Error 429")
}

func never() bool { return false }
