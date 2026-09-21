package approval

import (
	"errors"
	"slices"
	"sort"
	"time"
)

// Outcome は Authorize の結果。
type Outcome string

const (
	Allow         Outcome = "allow"
	NeedsApproval Outcome = "pending"
	Deny          Outcome = "forbidden"
)

// Decision は Authorize の結果と、承認待ちなら依頼。
type Decision struct {
	Outcome Outcome
	Risk    Risk
	Request *Request
}

// ErrNotYours は他人の依頼を実行しようとしたことを示す。
var ErrNotYours = errors.New("approval: 依頼者本人しか実行できない")

// ErrNotApproved は承認されていない依頼を実行しようとしたことを示す。
var ErrNotApproved = errors.New("approval: 承認されていない")

// AuditEntry は監査ログの 1 行。
type AuditEntry struct {
	RequestID string
	Action    string
	Actor     string
	At        time.Time
	Detail    string
}

// Authorize はリスクを判定し、自動で実行してよいか、承認を待つか、拒否するかを返す。
// 承認を待つなら依頼を作る。判定はこのサービスが持つルールで行い、呼び出し側に決めさせない。
func (s *Service) Authorize(requester, tool string, args map[string]any) (Decision, error) {
	risk := s.Assess(tool, args)
	switch risk {
	case Low:
		s.audit("", "auto_allow", requester, tool)
		return Decision{Outcome: Allow, Risk: risk}, nil
	case Forbidden:
		s.audit("", "forbidden", requester, tool)
		return Decision{Outcome: Deny, Risk: risk}, nil
	case Medium, High:
	}
	r, err := s.Open(requester, tool, args, risk)
	if err != nil {
		return Decision{}, err
	}
	return Decision{Outcome: NeedsApproval, Risk: risk, Request: r}, nil
}

// Take は承認済みの依頼を依頼者本人に渡し、使用済みにする。
// 実行する側は、ここで受け取った引数で実行する。モデルが後から出した引数は使わない。
func (s *Service) Take(caller, id string) (Request, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	r, ok := s.requests[id]
	switch {
	case !ok:
		return Request{}, ErrNotFound
	case r.Requester != caller:
		s.auditLocked(id, "take_denied", caller, "依頼者以外")
		return Request{}, ErrNotYours
	case r.Status != Approved:
		return Request{}, ErrNotApproved
	case !time.Now().Before(r.ExpiresAt):
		return Request{}, ErrExpired
	}
	r.Status = Used
	s.auditLocked(id, "take", caller, r.Tool)
	return *r, nil
}

// Filter は List の絞り込み。
type Filter int

const (
	// Mine は自分が依頼したもの。
	Mine Filter = iota
	// ToApprove は自分が判断できる承認待ち。高リスクは承認者だけに、中リスクは依頼者本人に見える。
	ToApprove
)

// List は依頼を作られた順に返す。
func (s *Service) List(caller string, f Filter) []Request {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out []Request
	for _, r := range s.requests {
		switch f {
		case Mine:
			if r.Requester != caller {
				continue
			}
		case ToApprove:
			if r.Status != Pending || !s.canDecide(caller, r) {
				continue
			}
		}
		out = append(out, *r)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].CreatedAt.Before(out[j].CreatedAt) })
	return out
}

func (s *Service) canDecide(caller string, r *Request) bool {
	switch r.Risk {
	case Medium:
		return caller == r.Requester
	case High:
		return caller != r.Requester && slices.Contains(s.approvers, caller)
	case Low, Forbidden:
	}
	return false
}

// Audit は依頼の監査ログを返す。依頼者と、その依頼を判断できる人だけが読める。
func (s *Service) Audit(caller, id string) ([]AuditEntry, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	r, ok := s.requests[id]
	if !ok {
		return nil, ErrNotFound
	}
	if r.Requester != caller && !s.canDecide(caller, r) && !slices.Contains(s.approvers, caller) {
		return nil, ErrNotApprover
	}
	var out []AuditEntry
	for _, e := range s.log {
		if e.RequestID == id {
			out = append(out, e)
		}
	}
	return out, nil
}

func (s *Service) audit(id, action, actor, detail string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.auditLocked(id, action, actor, detail)
}

func (s *Service) auditLocked(id, action, actor, detail string) {
	s.log = append(s.log, AuditEntry{RequestID: id, Action: action, Actor: actor, At: time.Now(), Detail: detail})
}
