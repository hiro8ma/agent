package agentmemory

import (
	"sort"
	"sync"
	"time"
)

// Store は Fact の保管庫。書き込みと読み出しの双方で検証する。
//
// 書き込み時だけの検証では、書いた時点で正しかった事実が
// 後から誤りになる場合を捕まえられない。
type Store struct {
	mu    sync.Mutex
	facts []Fact

	writeRules []Rule
	readRules  []Rule

	// Rejected は書き込みを拒んだ件数。
	Rejected int
	// Filtered は読み出しで除いた延べ件数。
	Filtered int
}

// NewStore は検証規則を与えて保管庫を作る。
func NewStore(write, read []Rule) *Store {
	return &Store{writeRules: write, readRules: read}
}

// Write は検証を通った Fact だけを保存する。
func (s *Store) Write(f Fact) error {
	for _, r := range s.writeRules {
		if err := r(f); err != nil {
			s.mu.Lock()
			s.Rejected++
			s.mu.Unlock()
			return err
		}
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.facts = append(s.facts, f)
	return nil
}

// Recall は主語に一致し、読み出し検証を通った Fact を返す。
//
// 同じ主語に複数あれば新しい順。矛盾は新しいものを優先する。
func (s *Store) Recall(subject string) []Fact {
	s.mu.Lock()
	defer s.mu.Unlock()

	var out []Fact
	for _, f := range s.facts {
		if f.Subject != subject {
			continue
		}
		ok := true
		for _, r := range s.readRules {
			if r(f) != nil {
				s.Filtered++
				ok = false
				break
			}
		}
		if ok {
			out = append(out, f)
		}
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].WrittenAt.After(out[j].WrittenAt) })
	return out
}

// Contradictions は同じ主語で内容の違う Fact の組を返す。
//
// 矛盾は書き込み時には分からない。後から入る Fact と衝突する。
func (s *Store) Contradictions() map[string][]Fact {
	s.mu.Lock()
	defer s.mu.Unlock()

	bySubject := map[string][]Fact{}
	for _, f := range s.facts {
		bySubject[f.Subject] = append(bySubject[f.Subject], f)
	}
	out := map[string][]Fact{}
	for subj, fs := range bySubject {
		seen := map[string]bool{}
		for _, f := range fs {
			seen[f.Statement] = true
		}
		if len(seen) > 1 {
			out[subj] = fs
		}
	}
	return out
}

// Len は保存件数。
func (s *Store) Len() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.facts)
}

// Now は試験で時刻を差し替えるための関数値。
var Now = time.Now
