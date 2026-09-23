// Package libbudget はトークン消費の上限を、セッション単位と全体の 2 段で数える。
package libbudget

import (
	"fmt"
	"os"
	"strconv"
	"sync"
)

// Limits はトークン消費の上限。0 は無制限。
type Limits struct {
	SessionTokens int
	TotalTokens   int
}

func (l Limits) Enabled() bool {
	return l.SessionTokens > 0 || l.TotalTokens > 0
}

// LimitsFromEnv は BUDGET_SESSION_TOKENS / BUDGET_TOTAL_TOKENS を読む。
// 未設定・数値でない・負数はいずれも無制限（0）として扱う。
func LimitsFromEnv() Limits {
	return Limits{
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

const (
	ScopeSession = "session"
	ScopeTotal   = "total"
)

type ExceededError struct {
	Scope     string
	SessionID string
	Limit     int
	Used      int
}

func (e *ExceededError) Error() string {
	if e.Scope == ScopeSession {
		return fmt.Sprintf("token budget exceeded: session %s used %d tokens of limit %d", e.SessionID, e.Used, e.Limit)
	}
	return fmt.Sprintf("token budget exceeded: total used %d tokens of limit %d", e.Used, e.Limit)
}

// Tracker の消費量は起動中のみ保持し、再起動でリセットされる。nil の Tracker は上限なしとして振る舞う。
type Tracker struct {
	limits Limits

	mu       sync.Mutex
	total    int
	sessions map[string]int
}

func NewTracker(limits Limits) *Tracker {
	return &Tracker{limits: limits, sessions: map[string]int{}}
}

// Check は消費前の残量確認。すでに上限に達していれば *ExceededError を返す。
// 1 回の応答で使うトークン数は事前にわからないため、上限は超過を検知した次の呼び出しで止まる。
func (b *Tracker) Check(sessionID string) error {
	if b == nil {
		return nil
	}
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.limits.SessionTokens > 0 && b.sessions[sessionID] >= b.limits.SessionTokens {
		return &ExceededError{Scope: ScopeSession, SessionID: sessionID, Limit: b.limits.SessionTokens, Used: b.sessions[sessionID]}
	}
	if b.limits.TotalTokens > 0 && b.total >= b.limits.TotalTokens {
		return &ExceededError{Scope: ScopeTotal, Limit: b.limits.TotalTokens, Used: b.total}
	}
	return nil
}

func (b *Tracker) Add(sessionID string, tokens int) {
	if b == nil || tokens <= 0 {
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	b.sessions[sessionID] += tokens
	b.total += tokens
}

func (b *Tracker) Used(sessionID string) (session, total int) {
	if b == nil {
		return 0, 0
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.sessions[sessionID], b.total
}
