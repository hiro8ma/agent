package agentcore

import (
	"fmt"
	"os"
	"strconv"
	"sync"
)

// BudgetLimits はトークン消費の上限。0 は無制限。
type BudgetLimits struct {
	SessionTokens int
	TotalTokens   int
}

// Enabled はどちらかの上限が設定されているかを返す。
func (l BudgetLimits) Enabled() bool {
	return l.SessionTokens > 0 || l.TotalTokens > 0
}

// BudgetLimitsFromEnv は BUDGET_SESSION_TOKENS / BUDGET_TOTAL_TOKENS を読む。
// 未設定・数値でない・負数はいずれも無制限（0）として扱う。
func BudgetLimitsFromEnv() BudgetLimits {
	return BudgetLimits{
		SessionTokens: envTokens("BUDGET_SESSION_TOKENS"),
		TotalTokens:   envTokens("BUDGET_TOTAL_TOKENS"),
	}
}

func envTokens(key string) int {
	n, err := strconv.Atoi(os.Getenv(key))
	if err != nil || n < 0 {
		return 0
	}
	return n
}

// BudgetError は上限超過。Connect ハンドラは resource_exhausted で返す。
type BudgetError struct {
	Scope     string // "session" | "total"
	SessionID string
	Limit     int
	Used      int
}

func (e *BudgetError) Error() string {
	if e.Scope == "session" {
		return fmt.Sprintf("token budget exceeded: session %s used %d tokens of limit %d", e.SessionID, e.Used, e.Limit)
	}
	return fmt.Sprintf("token budget exceeded: total used %d tokens of limit %d", e.Used, e.Limit)
}

// BudgetTracker はプロセス内のトークン消費量をセッション単位と全体の 2 段で数える。
// 消費量は起動中のみ保持し、再起動でリセットされる。
type BudgetTracker struct {
	limits BudgetLimits

	mu       sync.Mutex
	total    int
	sessions map[string]int
}

func NewBudgetTracker(limits BudgetLimits) *BudgetTracker {
	return &BudgetTracker{limits: limits, sessions: map[string]int{}}
}

// Check は消費前の残量確認。すでに上限に達していれば *BudgetError を返す。
// 1 回の応答で使うトークン数は事前にわからないため、上限は超過を検知した次の呼び出しで止まる。
func (b *BudgetTracker) Check(sessionID string) error {
	if b == nil {
		return nil
	}
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.limits.SessionTokens > 0 && b.sessions[sessionID] >= b.limits.SessionTokens {
		return &BudgetError{Scope: "session", SessionID: sessionID, Limit: b.limits.SessionTokens, Used: b.sessions[sessionID]}
	}
	if b.limits.TotalTokens > 0 && b.total >= b.limits.TotalTokens {
		return &BudgetError{Scope: "total", Limit: b.limits.TotalTokens, Used: b.total}
	}
	return nil
}

// Add は 1 回の応答の消費量を加算する。
func (b *BudgetTracker) Add(sessionID string, usage TokenUsage) {
	if b == nil {
		return
	}
	n := usage.TotalTokens
	if n <= 0 {
		n = usage.InputTokens + usage.OutputTokens
	}
	if n <= 0 {
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	b.sessions[sessionID] += n
	b.total += n
}

// Used はログ用の現在の消費量。
func (b *BudgetTracker) Used(sessionID string) (session, total int) {
	if b == nil {
		return 0, 0
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.sessions[sessionID], b.total
}
