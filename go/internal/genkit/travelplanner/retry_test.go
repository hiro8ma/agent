package travelplanner

import (
	"errors"
	"testing"
	"time"
)

func TestWithRetry(t *testing.T) {
	t.Parallel()
	errQuota := errors.New("Error 429, Status: RESOURCE_EXHAUSTED")
	errBad := errors.New("Error 400, Status: INVALID_ARGUMENT")
	testCases := map[string]struct {
		errs      []error
		streamed  bool
		wantErr   error
		wantCalls int
	}{
		"429が1回だけなら呼び直して成功する":  {errs: []error{errQuota, nil}, wantCalls: 2},
		"429が続けば上限の2回で諦める":     {errs: []error{errQuota, errQuota, errQuota, nil}, wantErr: errQuota, wantCalls: 3},
		"429以外のエラーは呼び直さない":     {errs: []error{errBad, nil}, wantErr: errBad, wantCalls: 1},
		"文章を送り始めた後の429は呼び直さない": {errs: []error{errQuota, nil}, streamed: true, wantErr: errQuota, wantCalls: 1},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			p := retryPolicy{maxRetries: 2, base: time.Millisecond, limit: time.Millisecond}
			calls := 0
			_, err := withRetry(t.Context(), p, func() bool { return tc.streamed }, func() (string, error) {
				err := tc.errs[calls]
				calls++
				return "", err
			})
			if !errors.Is(err, tc.wantErr) {
				t.Fatalf("err = %v, want %v", err, tc.wantErr)
			}
			if calls != tc.wantCalls {
				t.Errorf("calls = %d, want %d", calls, tc.wantCalls)
			}
		})
	}
}
