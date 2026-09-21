// Package expense は経費精算のエージェント。評価、ガードレール、HITL を組み合わせる。
package expense

import (
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"
)

// Status は経費の状態。
type Status string

const (
	Submitted       Status = "submitted"
	PendingApproval Status = "pending_approval"
	Approved        Status = "approved"
	Rejected        Status = "rejected"
)

// Expense は 1 件の経費。
type Expense struct {
	ID          string `json:"id"`
	UserID      string `json:"userId"`
	Date        string `json:"date"`
	Category    string `json:"category"`
	Amount      int    `json:"amount"`
	Description string `json:"description"`
	Status      Status `json:"status"`
	RequestID   string `json:"requestId,omitempty"`
}

// Store は経費の置き場。
type Store struct {
	mu       sync.Mutex
	next     int
	expenses map[string]*Expense
}

// NewStore は決まった経費を持った置き場を返す。照会の確認に使う。
func NewStore() *Store {
	s := &Store{expenses: map[string]*Expense{}}
	s.add(Expense{UserID: "user-001", Date: "2025-07-03", Category: "交通費", Amount: 1200, Description: "客先訪問の移動", Status: Submitted})
	s.add(Expense{UserID: "user-001", Date: "2025-07-18", Category: "会議費", Amount: 4800, Description: "取引先との打ち合わせ", Status: Submitted})
	s.add(Expense{UserID: "user-002", Date: "2025-07-05", Category: "交通費", Amount: 900, Description: "社外研修の移動", Status: Submitted})
	return s
}

func (s *Store) add(e Expense) *Expense {
	s.next++
	e.ID = fmt.Sprintf("exp-%03d", s.next)
	s.expenses[e.ID] = &e
	return &e
}

// Add は経費を登録し、ID を振って返す。
func (s *Store) Add(e Expense) Expense {
	s.mu.Lock()
	defer s.mu.Unlock()
	return *s.add(e)
}

// Get は経費を返す。
func (s *Store) Get(id string) (Expense, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	e, ok := s.expenses[id]
	if !ok {
		return Expense{}, false
	}
	return *e, true
}

// SetStatus は経費の状態を変える。
func (s *Store) SetStatus(id string, st Status) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if e, ok := s.expenses[id]; ok {
		e.Status = st
	}
}

// List は利用者の経費を日付の順に返す。month（2025-07 の形）が空でなければその月に絞る。
func (s *Store) List(userID, month string) []Expense {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out []Expense
	for _, e := range s.expenses {
		if e.UserID == userID && (month == "" || strings.HasPrefix(e.Date, month+"-")) {
			out = append(out, *e)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Date < out[j].Date })
	return out
}

// ValidDate は 2025-07-10 の形の実在する日付かを返す。
func ValidDate(s string) bool {
	_, err := time.Parse(time.DateOnly, s)
	return err == nil
}

func (s *Store) setRequest(id, requestID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if e, ok := s.expenses[id]; ok {
		e.RequestID = requestID
	}
}
