package agentcore

import (
	"errors"
	"strings"
	"sync"
	"testing"
)

func TestBudgetTracker_SessionLimit(t *testing.T) {
	b := NewBudgetTracker(BudgetLimits{SessionTokens: 100})

	if err := b.Check("s1"); err != nil {
		t.Fatalf("first check: %v", err)
	}
	b.Add("s1", TokenUsage{InputTokens: 60, OutputTokens: 50, TotalTokens: 110})

	err := b.Check("s1")
	var budgetErr *BudgetError
	if !errors.As(err, &budgetErr) {
		t.Fatalf("want *BudgetError, got %v", err)
	}
	if budgetErr.Scope != "session" || budgetErr.Used != 110 || budgetErr.Limit != 100 {
		t.Fatalf("unexpected error detail: %+v", budgetErr)
	}
	if !strings.Contains(budgetErr.Error(), "s1") {
		t.Fatalf("message should name the session: %s", budgetErr.Error())
	}

	// 別セッションは自分の消費量だけを見る。
	if err := b.Check("s2"); err != nil {
		t.Fatalf("other session: %v", err)
	}
}

func TestBudgetTracker_TotalLimit(t *testing.T) {
	b := NewBudgetTracker(BudgetLimits{TotalTokens: 100})

	b.Add("s1", TokenUsage{TotalTokens: 60})
	if err := b.Check("s2"); err != nil {
		t.Fatalf("under total limit: %v", err)
	}

	b.Add("s2", TokenUsage{TotalTokens: 60})
	var budgetErr *BudgetError
	if err := b.Check("s3"); !errors.As(err, &budgetErr) {
		t.Fatalf("want *BudgetError, got %v", err)
	}
	if budgetErr.Scope != "total" || budgetErr.Used != 120 {
		t.Fatalf("unexpected error detail: %+v", budgetErr)
	}
}

func TestBudgetTracker_ZeroIsUnlimited(t *testing.T) {
	b := NewBudgetTracker(BudgetLimits{})
	b.Add("s1", TokenUsage{TotalTokens: 1_000_000})
	if err := b.Check("s1"); err != nil {
		t.Fatalf("zero limits must not reject: %v", err)
	}

	// nil トラッカー（予算未設定）も従来どおり通す。
	var nilTracker *BudgetTracker
	nilTracker.Add("s1", TokenUsage{TotalTokens: 10})
	if err := nilTracker.Check("s1"); err != nil {
		t.Fatalf("nil tracker must not reject: %v", err)
	}
	if session, total := nilTracker.Used("s1"); session != 0 || total != 0 {
		t.Fatalf("nil tracker used: session=%d total=%d", session, total)
	}
}

func TestBudgetTracker_ConcurrentAdd(t *testing.T) {
	b := NewBudgetTracker(BudgetLimits{SessionTokens: 1_000_000, TotalTokens: 1_000_000})

	const goroutines, perGoroutine = 50, 100
	var wg sync.WaitGroup
	for i := range goroutines {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			sessionID := []string{"s1", "s2"}[i%2]
			for range perGoroutine {
				b.Add(sessionID, TokenUsage{InputTokens: 1, OutputTokens: 1})
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

func TestBudgetLimitsFromEnv(t *testing.T) {
	t.Setenv("BUDGET_SESSION_TOKENS", "5000")
	t.Setenv("BUDGET_TOTAL_TOKENS", "abc")
	limits := BudgetLimitsFromEnv()
	if limits.SessionTokens != 5000 || limits.TotalTokens != 0 {
		t.Fatalf("unexpected limits: %+v", limits)
	}
	if !limits.Enabled() {
		t.Fatal("session limit alone should enable the budget")
	}
	if (BudgetLimits{}).Enabled() {
		t.Fatal("zero limits must be disabled")
	}
}
