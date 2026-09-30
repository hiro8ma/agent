package fst

import (
	"slices"
	"strconv"
	"strings"
)

// nstate は非決定性のレーベンシュタインオートマトンの状態。i はクエリを何文字読んだか、e はここまでの編集の回数。
type nstate struct{ i, e int }

// Levenshtein はクエリから距離 k 以内の語だけを受理する決定性オートマトン（DFA）。
// 非決定性の状態の集合を 1 つの状態とみなし、必要になった遷移だけを作ってためておく。
type Levenshtein struct {
	q      []rune
	k      int
	start  *dstate
	states map[string]*dstate
}

type dstate struct {
	set   []nstate
	next  map[rune]*dstate
	other *dstate
	// dist は受理するときの最小の距離で、受理しないなら -1。
	dist int
	dead bool
}

func NewLevenshtein(query string, k int) *Levenshtein {
	l := &Levenshtein{q: []rune(query), k: k, states: make(map[string]*dstate)}
	l.start = l.intern(l.closure([]nstate{{0, 0}}))
	return l
}

// closure はクエリの文字を読み飛ばす削除（入力を消費しない遷移）をたどり、同じ位置でより多い編集の状態を落とす。
func (l *Levenshtein) closure(set []nstate) []nstate {
	best := make(map[int]int)
	stack := slices.Clone(set)
	for len(stack) > 0 {
		s := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if s.e > l.k {
			continue
		}
		if e, ok := best[s.i]; ok && e <= s.e {
			continue
		}
		best[s.i] = s.e
		if s.i < len(l.q) {
			stack = append(stack, nstate{s.i + 1, s.e + 1})
		}
	}
	out := make([]nstate, 0, len(best))
	for i, e := range best {
		out = append(out, nstate{i, e})
	}
	slices.SortFunc(out, func(a, b nstate) int { return a.i - b.i })
	return out
}

func (l *Levenshtein) intern(set []nstate) *dstate {
	var key strings.Builder
	for _, s := range set {
		key.WriteString(strconv.Itoa(s.i))
		key.WriteByte(':')
		key.WriteString(strconv.Itoa(s.e))
		key.WriteByte(',')
	}
	if d, ok := l.states[key.String()]; ok {
		return d
	}
	d := &dstate{set: set, next: make(map[rune]*dstate), dist: -1, dead: len(set) == 0}
	for _, s := range set {
		if s.i == len(l.q) && (d.dist < 0 || s.e < d.dist) {
			d.dist = s.e
		}
	}
	l.states[key.String()] = d
	return d
}

// step は c を読んだ後の状態を返す。クエリに出てこない文字はどれも同じ遷移になるので other にまとめる。
func (l *Levenshtein) step(d *dstate, c rune) *dstate {
	inQuery := false
	for _, s := range d.set {
		if s.i < len(l.q) && l.q[s.i] == c {
			inQuery = true
			break
		}
	}
	if inQuery {
		if n, ok := d.next[c]; ok {
			return n
		}
	} else if d.other != nil {
		return d.other
	}
	var set []nstate
	for _, s := range d.set {
		if s.i < len(l.q) && l.q[s.i] == c {
			set = append(set, nstate{s.i + 1, s.e})
		}
		set = append(set, nstate{s.i, s.e + 1}) // 挿入
		if s.i < len(l.q) {
			set = append(set, nstate{s.i + 1, s.e + 1}) // 置換
		}
	}
	n := l.intern(l.closure(set))
	if inQuery {
		d.next[c] = n
	} else {
		d.other = n
	}
	return n
}

// States は今までに作った DFA の状態の数を返す。
func (l *Levenshtein) States() int { return len(l.states) }

// Match は語と、クエリからの距離。
type Match struct {
	Entry
	Dist int
}

// Fuzzy は FST と DFA を同時にたどり、距離 k 以内の語を返す。DFA が死んだ枝はそこで打ち切る。
func (f *FST) Fuzzy(l *Levenshtein) []Match {
	var out []Match
	var buf []rune
	var walk func(s int, d *dstate, sum uint64)
	walk = func(s int, d *dstate, sum uint64) {
		n := &f.nodes[s]
		if n.final && d.dist >= 0 {
			out = append(out, Match{Key: string(buf), Out: sum + n.finalOut, Dist: d.dist})
		}
		for _, t := range n.trans {
			nd := l.step(d, t.label)
			if nd.dead {
				continue
			}
			buf = append(buf, t.label)
			walk(t.to, nd, sum+t.out)
			buf = buf[:len(buf)-1]
		}
	}
	walk(f.root, l.start, 0)
	return out
}

// FuzzyRows は DFA を作らず、状態ごとに動的計画法の 1 行を持ってたどる。行の最小が k を超えたら打ち切る。
func (f *FST) FuzzyRows(query string, k int) []Match {
	q := []rune(query)
	row := make([]int, len(q)+1)
	for i := range row {
		row[i] = i
	}
	var out []Match
	var buf []rune
	var walk func(s int, row []int, sum uint64)
	walk = func(s int, row []int, sum uint64) {
		n := &f.nodes[s]
		if n.final && row[len(q)] <= k {
			out = append(out, Match{Key: string(buf), Out: sum + n.finalOut, Dist: row[len(q)]})
		}
		for _, t := range n.trans {
			next := make([]int, len(q)+1)
			next[0] = row[0] + 1
			best := next[0]
			for i := 1; i <= len(q); i++ {
				cost := 1
				if q[i-1] == t.label {
					cost = 0
				}
				next[i] = min(row[i]+1, next[i-1]+1, row[i-1]+cost)
				best = min(best, next[i])
			}
			if best > k {
				continue
			}
			buf = append(buf, t.label)
			walk(t.to, next, sum+t.out)
			buf = buf[:len(buf)-1]
		}
	}
	walk(f.root, row, 0)
	return out
}

// Distance はレーベンシュタイン距離（挿入、削除、置換）を動的計画法で O(MN) で求める。
func Distance(a, b string) int {
	s, t := []rune(a), []rune(b)
	prev := make([]int, len(t)+1)
	cur := make([]int, len(t)+1)
	for j := range prev {
		prev[j] = j
	}
	for i := 1; i <= len(s); i++ {
		cur[0] = i
		for j := 1; j <= len(t); j++ {
			cost := 1
			if s[i-1] == t[j-1] {
				cost = 0
			}
			cur[j] = min(prev[j]+1, cur[j-1]+1, prev[j-1]+cost)
		}
		prev, cur = cur, prev
	}
	return prev[len(t)]
}
