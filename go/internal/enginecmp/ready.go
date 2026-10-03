package enginecmp

import (
	"context"
	"errors"
	"fmt"
	"time"
)

// DefaultQueries は形態素と bigram、BM25 と Meilisearch の規則の違いが順位に出るクエリ。
var DefaultQueries = []string{
	"すべて国民",
	"侵さない",
	"会議",
	"本国",
	"表現の自由",
	"天皇 国会 召集",
}

// WaitReady は checks がすべて成功するまで、timeout を上限に 1 秒おきに試す。
func WaitReady(ctx context.Context, timeout time.Duration, checks ...func(context.Context) error) error {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	for {
		var errs []error
		for _, check := range checks {
			if err := check(ctx); err != nil {
				errs = append(errs, err)
			}
		}
		if len(errs) == 0 {
			return nil
		}
		select {
		case <-ctx.Done():
			return fmt.Errorf("起動を待ちきれなかった: %w", errors.Join(errs...))
		case <-ticker.C:
		}
	}
}
