package libbudget

import (
	"errors"
	"strings"
	"sync"
	"testing"
)

func TestTracker_SessionLimit(t *testing.T) {
	b := NewTracker(Limits{SessionTokens: 100})

	if err := b.Check("s1"); err != nil {
		t.Fatalf("first check: %v", err)
	}
	b.Add("s1", 110)

	err := b.Check("s1")
	var budgetErr *ExceededError
	if !errors.As(err, &budgetErr) {
		t.Fatalf("want *ExceededError, got %v", err)
	}
	if budgetErr.Scope != ScopeSession || budgetErr.Used != 110 || budgetErr.Limit != 100 {
		t.Fatalf("unexpected error detail: %+v", budgetErr)
	}
	if !strings.Contains(budgetErr.Error(), "s1") {
		t.Fatalf("message should name the session: %s", budgetErr.Error())
	}

	if err := b.Check("s2"); err != nil {
		t.Fatalf("other session: %v", err)
	}
}

func TestTracker_TotalLimit(t *testing.T) {
	b := NewTracker(Limits{TotalTokens: 100})

	b.Add("s1", 60)
	if err := b.Check("s2"); err != nil {
		t.Fatalf("under total limit: %v", err)
	}

	b.Add("s2", 60)
	var budgetErr *ExceededError
	if err := b.Check("s3"); !errors.As(err, &budgetErr) {
		t.Fatalf("want *ExceededError, got %v", err)
	}
	if budgetErr.Scope != ScopeTotal || budgetErr.Used != 120 {
		t.Fatalf("unexpected error detail: %+v", budgetErr)
	}
}

func TestTracker_ZeroIsUnlimited(t *testing.T) {
	b := NewTracker(Limits{})
	b.Add("s1", 1_000_000)
	if err := b.Check("s1"); err != nil {
		t.Fatalf("zero limits must not reject: %v", err)
	}

	var nilTracker *Tracker
	nilTracker.Add("s1", 10)
	if err := nilTracker.Check("s1"); err != nil {
		t.Fatalf("nil tracker must not reject: %v", err)
	}
	if session, total := nilTracker.Used("s1"); session != 0 || total != 0 {
		t.Fatalf("nil tracker used: session=%d total=%d", session, total)
	}
}

func TestTracker_IgnoresNonPositive(t *testing.T) {
	b := NewTracker(Limits{TotalTokens: 1})
	b.Add("s1", 0)
	b.Add("s1", -5)
	if session, total := b.Used("s1"); session != 0 || total != 0 {
		t.Fatalf("used: session=%d total=%d, want 0 0", session, total)
	}
}

func TestTracker_ConcurrentAdd(t *testing.T) {
	b := NewTracker(Limits{SessionTokens: 1_000_000, TotalTokens: 1_000_000})

	const goroutines, perGoroutine = 50, 100
	var wg sync.WaitGroup
	for i := range goroutines {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			sessionID := []string{"s1", "s2"}[i%2]
			for range perGoroutine {
				b.Add(sessionID, 2)
				_ = b.Check(sessionID)
				b.Used(sessionID)
			}
		}(i)
	}
	wg.Wait()

	wantTotal := goroutines * perGoroutine * 2
	s1, total := b.Used("s1")
	s2, _ := b.Used("s2")
	if total != wantTotal {
		t.Fatalf("total: want %d, got %d", wantTotal, total)
	}
	if s1+s2 != wantTotal {
		t.Fatalf("per session sum: want %d, got %d", wantTotal, s1+s2)
	}
}

func TestLimitsFromEnv(t *testing.T) {
	t.Setenv("BUDGET_SESSION_TOKENS", "5000")
	t.Setenv("BUDGET_TOTAL_TOKENS", "abc")
	limits := LimitsFromEnv()
	if limits.SessionTokens != 5000 || limits.TotalTokens != 0 {
		t.Fatalf("unexpected limits: %+v", limits)
	}
	if !limits.Enabled() {
		t.Fatal("session limit alone should enable the budget")
	}
	if (Limits{}).Enabled() {
		t.Fatal("zero limits must be disabled")
	}
}
