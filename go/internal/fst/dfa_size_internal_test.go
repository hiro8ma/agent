package fst

import (
	"math/rand/v2"
	"strconv"
	"strings"
	"testing"
)

// subsume は (i,e) より少ない編集 e' で |i-i'| <= e-e' の (i',e') があれば (i,e) を落とす。
// (i',e') から e-e' 回の挿入か削除で (i,e) の受理する語をすべて受理できるので、受理する語は変わらない（Schulz と Mihov の包含）。
func subsume(set []nstate) []nstate {
	var out []nstate
	for _, a := range set {
		covered := false
		for _, b := range set {
			if b.e < a.e && abs(a.i-b.i) <= a.e-b.e {
				covered = true
				break
			}
		}
		if !covered {
			out = append(out, a)
		}
	}
	return out
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func setKey(set []nstate) string {
	var b strings.Builder
	for _, s := range set {
		b.WriteString(strconv.Itoa(s.i))
		b.WriteByte(':')
		b.WriteString(strconv.Itoa(s.e))
		b.WriteByte(',')
	}
	return b.String()
}

// countReduced は reduce をかけた集合を状態の名前にして、空集合を除くたどり着ける状態を数える。
// 包含で落とした (i,e) には削除でたどる先も含まれるので、遷移は名前の集合を閉包し直してから求める。
func countReduced(query string, k int, reduce func([]nstate) []nstate) int {
	l := NewLevenshtein(query, k)
	alphabet := append(distinctRunes(l.q), otherRune)
	start := reduce(l.start.set)
	seen := map[string]bool{setKey(start): true}
	queue := [][]nstate{start}
	for len(queue) > 0 {
		set := queue[0]
		queue = queue[1:]
		for _, c := range alphabet {
			n := reduce(l.successor(l.closure(set), c))
			if len(n) == 0 {
				continue
			}
			if key := setKey(n); !seen[key] {
				seen[key] = true
				queue = append(queue, n)
			}
		}
	}
	return len(seen)
}

// minimalStates は受理するかどうかで分けた状態を、遷移先の組が同じものだけ残るまで分け直し（Moore の方法）、最小の DFA の状態の数を返す。空集合は含めない。
func minimalStates(d *DFA) int {
	class := make([]int, len(d.sets))
	for s, e := range d.dist {
		if e >= 0 {
			class[s] = 1
		}
	}
	count := -1
	for {
		ids := make(map[string]int)
		next := make([]int, len(d.sets))
		for s := range d.sets {
			var b strings.Builder
			b.WriteString(strconv.Itoa(class[s]))
			for _, to := range d.next[s] {
				b.WriteByte(',')
				if to < 0 {
					b.WriteString("x")
				} else {
					b.WriteString(strconv.Itoa(class[to]))
				}
			}
			key := b.String()
			id, ok := ids[key]
			if !ok {
				id = len(ids)
				ids[key] = id
			}
			next[s] = id
		}
		class = next
		if len(ids) == count {
			return count
		}
		count = len(ids)
	}
}

func randomQuery(r *rand.Rand, n int, alphabet string) string {
	a := []rune(alphabet)
	q := make([]rune, n)
	for i := range q {
		q[i] = a[r.IntN(len(a))]
	}
	return string(q)
}

func repeat(pattern string, n int) string {
	p := []rune(pattern)
	q := make([]rune, n)
	for i := range q {
		q[i] = p[i%len(p)]
	}
	return string(q)
}

// TestDFAStateBounds は n=1..30 の繰り返しの文字列と乱数の文字列で状態の数を数え、n ごとの最大が教材の上限 12(n+2)（k=1）と 144(n+3)（k=2）に収まることを確かめる。
// 最大を取ったクエリでは、最小の DFA の状態の数と、包含で縮めた状態の数も出す。
//
//	go test -run TestDFAStateBounds -v ./internal/fst/
func TestDFAStateBounds(t *testing.T) {
	t.Parallel()
	bound := map[int]func(n int) int{1: func(n int) int { return 12 * (n + 2) }, 2: func(n int) int { return 144 * (n + 3) }}
	r := rand.New(rand.NewPCG(1, 2))
	for _, k := range []int{1, 2} {
		for n := 1; n <= 30; n++ {
			queries := []string{repeat("a", n), repeat("ab", n), repeat("abc", n), repeat("aab", n), repeat("abcdefghijklmnopqrstuvwxyz0123456789", n)}
			for range 40 {
				queries = append(queries, randomQuery(r, n, "ab"), randomQuery(r, n, "abc"), randomQuery(r, n, "abcd"))
			}
			var worst *DFA
			worstQuery := ""
			for _, q := range queries {
				if d := NewDFA(q, k); worst == nil || d.States() > worst.States() {
					worst, worstQuery = d, q
				}
			}
			minimal, reduced := minimalStates(worst), countReduced(worstQuery, k, subsume)
			t.Logf("k=%d n=%2d 最大 %3d 最小 DFA %3d 包含 %3d 上限 %4d (%s)", k, n, worst.States(), minimal, reduced, bound[k](n), worstQuery)
			if worst.States() > bound[k](n) {
				t.Errorf("k=%d n=%d: %d 状態が上限 %d を超える (%s)", k, n, worst.States(), bound[k](n), worstQuery)
			}
			if minimal > worst.States() || reduced < minimal {
				t.Errorf("k=%d n=%d: 最小 %d / 包含 %d / 作った %d の大小が合わない", k, n, minimal, reduced, worst.States())
			}
		}
	}
}
