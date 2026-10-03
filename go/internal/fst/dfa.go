package fst

import (
	"slices"
	"unicode/utf8"
)

// otherRune はクエリに出てこない文字の代表。文字列から取り出した rune は負にならないので、クエリの文字と一致しない。
const otherRune rune = -1

// DFA はクエリと k から、たどり着けるすべての状態と遷移を先に作ったレーベンシュタインオートマトン。
// 遷移の列はクエリに出てくる文字を昇順に並べたものと、最後にクエリに出てこない文字（other）の 1 列。
// 空集合の状態（どの語も受理できない状態）は持たず、そこへの遷移は -1 にする。
type DFA struct {
	alphabet []rune
	sets     [][]nstate
	next     [][]int32
	dist     []int
}

// NewDFA は開始状態から幅優先に、各状態でアルファベットのすべての列の遷移を求める。状態の番号は見つけた順。
func NewDFA(query string, k int) *DFA {
	l := NewLevenshtein(query, k)
	d := &DFA{alphabet: distinctRunes(l.q)}
	if l.start.dead {
		return d
	}
	ids := map[*dstate]int32{l.start: 0}
	order := []*dstate{l.start}
	for s := 0; s < len(order); s++ {
		row := make([]int32, len(d.alphabet)+1)
		for c := range row {
			r := otherRune
			if c < len(d.alphabet) {
				r = d.alphabet[c]
			}
			n := l.step(order[s], r)
			if n.dead {
				row[c] = -1
				continue
			}
			id, ok := ids[n]
			if !ok {
				id = int32(len(order))
				ids[n] = id
				order = append(order, n)
			}
			row[c] = id
		}
		d.next = append(d.next, row)
	}
	for _, s := range order {
		d.sets = append(d.sets, s.set)
		d.dist = append(d.dist, s.dist)
	}
	return d
}

func distinctRunes(q []rune) []rune {
	out := slices.Clone(q)
	slices.Sort(out)
	return slices.Compact(out)
}

// States は空集合を除いた状態の数を返す。
func (d *DFA) States() int { return len(d.sets) }

// Transitions は空集合の状態へ行かない遷移の数を返す。
func (d *DFA) Transitions() int {
	n := 0
	for _, row := range d.next {
		for _, to := range row {
			if to >= 0 {
				n++
			}
		}
	}
	return n
}

// Accepting は受理する状態の数を返す。
func (d *DFA) Accepting() int {
	n := 0
	for _, e := range d.dist {
		if e >= 0 {
			n++
		}
	}
	return n
}

func (d *DFA) column(r rune) int {
	if i, ok := slices.BinarySearch(d.alphabet, r); ok {
		return i
	}
	return len(d.alphabet)
}

func (d *DFA) step(s int32, r rune) int32 { return d.next[s][d.column(r)] }

// Accept は word を受理すればクエリからの距離と true を返す。
func (d *DFA) Accept(word string) (int, bool) {
	if len(d.sets) == 0 {
		return 0, false
	}
	s := int32(0)
	for len(word) > 0 {
		r, n := utf8.DecodeRuneInString(word)
		word = word[n:]
		if s = d.step(s, r); s < 0 {
			return 0, false
		}
	}
	return d.dist[s], d.dist[s] >= 0
}

// FuzzyDFA は Fuzzy と同じく FST と DFA を同時にたどる。遷移は表を引くだけで、途中で状態を作らない。
func (f *FST) FuzzyDFA(d *DFA) []Match {
	if len(d.sets) == 0 {
		return nil
	}
	var out []Match
	var buf []rune
	var walk func(s int, q int32, sum uint64)
	walk = func(s int, q int32, sum uint64) {
		n := &f.nodes[s]
		if n.final && d.dist[q] >= 0 {
			out = append(out, Match{Key: string(buf), Out: sum + n.finalOut, Dist: d.dist[q]})
		}
		for _, t := range n.trans {
			nq := d.step(q, t.label)
			if nq < 0 {
				continue
			}
			buf = append(buf, t.label)
			walk(t.to, nq, sum+t.out)
			buf = buf[:len(buf)-1]
		}
	}
	walk(f.root, 0, 0)
	return out
}

// Fuzziness は語の長さ（rune の数）から許す距離を決める。One 文字以上で 1、Two 文字以上で 2 を許し、それより短い語は完全一致だけにする。
type Fuzziness struct {
	One, Two int
}

// DefaultFuzziness は 1 から 4 文字は完全一致、5 から 8 文字は距離 1、9 文字以上は距離 2。
var DefaultFuzziness = Fuzziness{One: 5, Two: 9}

func (f Fuzziness) K(word string) int {
	n := utf8.RuneCountInString(word)
	switch {
	case n >= f.Two:
		return 2
	case n >= f.One:
		return 1
	}
	return 0
}
