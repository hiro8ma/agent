// Package autonomy は、エージェントの操作の種類ごとに、任せる範囲を Read → Draft → Act の順に広げる。
// Draft（人が承認する提案）の承認率が直近の窓で基準を超えたら Act（自動実行）に上げ、Act で事故が起きたら Draft に戻して実績を数え直す。
package autonomy

import "sync"

// Level はエージェントに任せる範囲。
type Level int

const (
	Read  Level = iota // 検索、要約だけ
	Draft              // 提案し、人が承認してから実行する
	Act                // 人の承認なしで実行する
)

func (l Level) String() string {
	return [...]string{"read", "draft", "act"}[l]
}

// Policy は操作の種類ごとの規則。
type Policy struct {
	Max        Level   // 上限。返金のように自動にしない操作は Draft にする
	Window     int     // 承認率を数える直近の Draft の件数
	MinSamples int     // 昇格の判断に要る最低件数
	PromoteAt  float64 // この承認率以上で Act に上げる
}

type state struct {
	level    Level
	outcomes []bool // 直近の Draft が修正なしで承認されたか
}

// Ladder は操作の種類ごとの今の範囲を持つ。
type Ladder struct {
	mu       sync.Mutex
	policies map[string]Policy
	states   map[string]*state
}

func New(policies map[string]Policy) *Ladder {
	l := &Ladder{policies: policies, states: map[string]*state{}}
	for kind, p := range policies {
		l.states[kind] = &state{level: min(p.Max, Draft)}
	}
	return l
}

// Level は kind の今の範囲を返す。規則のない操作は Read にする。
func (l *Ladder) Level(kind string) Level {
	l.mu.Lock()
	defer l.mu.Unlock()
	s, ok := l.states[kind]
	if !ok {
		return Read
	}
	return s.level
}

// RecordDraft は Draft を人が修正なしで承認したか（approved）を記録し、基準を満たせば Act に上げる。
func (l *Ladder) RecordDraft(kind string, approved bool) {
	l.mu.Lock()
	defer l.mu.Unlock()
	s, ok := l.states[kind]
	if !ok || s.level != Draft {
		return
	}
	p := l.policies[kind]
	s.outcomes = append(s.outcomes, approved)
	if len(s.outcomes) > p.Window {
		s.outcomes = s.outcomes[len(s.outcomes)-p.Window:]
	}
	if p.Max >= Act && len(s.outcomes) >= p.MinSamples && rate(s.outcomes) >= p.PromoteAt {
		s.level = Act
	}
}

// RecordIncident は Act での事故（取り消し、苦情）を記録し、Draft に戻して実績を数え直す。
func (l *Ladder) RecordIncident(kind string) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if s, ok := l.states[kind]; ok && s.level == Act {
		s.level = Draft
		s.outcomes = nil
	}
}

func rate(outcomes []bool) float64 {
	n := 0
	for _, ok := range outcomes {
		if ok {
			n++
		}
	}
	return float64(n) / float64(len(outcomes))
}
