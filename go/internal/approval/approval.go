// Package approval はツール実行の前に人の承認を挟む。
//
// 承認待ちと承認済みを Session の State に置くと、2 つの抜け道ができる。
//
//	State は ADK の REST の /run が受ける stateDelta でクライアントが書ける。利用者が自分で承認済みにできる
//	承認が引数に結び付かない。少額で承認させた後に、同じ印で高額を実行できる
//
// ここでは承認をサーバ側の Store に持ち、ツール名と引数の組、依頼者、1 回限り、期限に結び付ける。
// 高リスクの承認は、依頼者と別の承認者だけができる。
package approval

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"sync"
	"time"
)

// Risk は操作のリスク。
type Risk int

const (
	// Low は自動で実行する。
	Low Risk = iota
	// Medium は依頼者本人の確認で実行する。
	Medium
	// High は依頼者と別の承認者の承認で実行する。
	High
	// Forbidden は承認があっても実行しない。
	Forbidden
)

func (r Risk) String() string {
	return [...]string{"low", "medium", "high", "forbidden"}[r]
}

// Rule はツールの引数からリスクを決める。
type Rule func(args map[string]any) Risk

// Status は承認依頼の状態。
type Status string

const (
	Pending  Status = "pending"
	Approved Status = "approved"
	Rejected Status = "rejected"
	Used     Status = "used"
)

// Request は 1 件の承認依頼。
type Request struct {
	ID        string
	Tool      string
	Args      map[string]any
	Digest    string
	Requester string
	Risk      Risk
	Status    Status
	Approver  string
	CreatedAt time.Time
	ExpiresAt time.Time
}

var (
	ErrNotFound      = errors.New("approval: 承認依頼が無い")
	ErrNotPending    = errors.New("approval: 承認待ちではない")
	ErrExpired       = errors.New("approval: 期限切れ")
	ErrSelfApproval  = errors.New("approval: 高リスクの操作は依頼者本人が承認できない")
	ErrNotApprover   = errors.New("approval: 承認する権限が無い")
	ErrNotRequester  = errors.New("approval: 中リスクの操作は依頼者本人が確認する")
	ErrForbiddenRisk = errors.New("approval: 承認の対象外の操作")
)

// Service は承認依頼を作り、承認し、実行の時に 1 回だけ消費する。
type Service struct {
	rules     map[string]Rule
	approvers []string
	ttl       time.Duration

	mu       sync.Mutex
	requests map[string]*Request
}

// NewService は tool 名ごとのルールと、高リスクを承認できる利用者と、承認の有効期限を受ける。
func NewService(rules map[string]Rule, approvers []string, ttl time.Duration) *Service {
	return &Service{rules: rules, approvers: approvers, ttl: ttl, requests: map[string]*Request{}}
}

// Assess はツールの呼び出しのリスクを返す。ルールの無いツールは Low。
func (s *Service) Assess(tool string, args map[string]any) Risk {
	rule, ok := s.rules[tool]
	if !ok {
		return Low
	}
	return rule(args)
}

// Digest はツール名と引数の組を 1 つの値にする。引数は JSON のキー順で並べる。
func Digest(tool string, args map[string]any) (string, error) {
	b, err := json.Marshal(args)
	if err != nil {
		return "", fmt.Errorf("approval: 引数の直列化: %w", err)
	}
	sum := sha256.Sum256(append([]byte(tool+"\x00"), b...))
	return hex.EncodeToString(sum[:]), nil
}

// Open は承認依頼を作る。
func (s *Service) Open(requester, tool string, args map[string]any, risk Risk) (*Request, error) {
	if risk == Forbidden {
		return nil, ErrForbiddenRisk
	}
	digest, err := Digest(tool, args)
	if err != nil {
		return nil, err
	}
	now := time.Now()
	r := &Request{
		ID: newID(), Tool: tool, Args: args, Digest: digest, Requester: requester, Risk: risk,
		Status: Pending, CreatedAt: now, ExpiresAt: now.Add(s.ttl),
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.requests[r.ID] = r
	return r, nil
}

// Decide は承認者の判断を記録する。
func (s *Service) Decide(approver, id string, approve bool) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	r, ok := s.requests[id]
	switch {
	case !ok:
		return ErrNotFound
	case r.Status != Pending:
		return ErrNotPending
	case !time.Now().Before(r.ExpiresAt):
		return ErrExpired
	case r.Risk == Medium && approver != r.Requester:
		return ErrNotRequester
	case r.Risk == High && approver == r.Requester:
		return ErrSelfApproval
	case r.Risk == High && !slices.Contains(s.approvers, approver):
		return ErrNotApprover
	}
	r.Approver = approver
	r.Status = Rejected
	if approve {
		r.Status = Approved
	}
	return nil
}

// Consume は同じ依頼者、同じツール名と引数の組で承認済みの依頼があれば、使用済みにして true を返す。
// 1 回使った承認は 2 回目には効かない。
func (s *Service) Consume(requester, tool string, args map[string]any) (bool, error) {
	digest, err := Digest(tool, args)
	if err != nil {
		return false, err
	}
	now := time.Now()
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, r := range s.requests {
		if r.Status == Approved && r.Digest == digest && r.Requester == requester && now.Before(r.ExpiresAt) {
			r.Status = Used
			return true, nil
		}
	}
	return false, nil
}

// Get は承認依頼を返す。
func (s *Service) Get(id string) (Request, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	r, ok := s.requests[id]
	if !ok {
		return Request{}, ErrNotFound
	}
	return *r, nil
}

func newID() string {
	b := make([]byte, 8)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}
