package fst

import (
	"fmt"
	"strings"
	"testing"
)

// draw は DFA の状態を番号順に 1 行ずつ、(i,j) の集合と列ごとの遷移で描く。* は受理する状態で、括弧の中は距離。other は * の列、∅ は空集合。
func (d *DFA) draw() string {
	var b strings.Builder
	for s, set := range d.sets {
		mark := " "
		if d.dist[s] >= 0 {
			mark = "*"
		}
		parts := make([]string, len(set))
		for i, n := range set {
			parts[i] = fmt.Sprintf("(%d,%d)", n.i, n.e)
		}
		fmt.Fprintf(&b, "%sq%d {%s}", mark, s, strings.Join(parts, " "))
		if d.dist[s] >= 0 {
			fmt.Fprintf(&b, " 距離%d", d.dist[s])
		}
		for c, to := range d.next[s] {
			label := "other"
			if c < len(d.alphabet) {
				label = string(d.alphabet[c])
			}
			target := "∅"
			if to >= 0 {
				target = fmt.Sprintf("q%d", to)
			}
			fmt.Fprintf(&b, " %s->%s", label, target)
		}
		b.WriteByte('\n')
	}
	return b.String()
}

// TestDrawDFAAbaaba は教材の例（abaaba、k=1）の DFA 全体を固定する。空集合を除いて 25 状態、受理する状態は 6。
func TestDrawDFAAbaaba(t *testing.T) {
	t.Parallel()
	d := NewDFA("abaaba", 1)
	want := ` q0 {(0,0) (1,1)} a->q1 b->q2 other->q3
 q1 {(0,1) (1,0) (2,1)} a->q4 b->q5 other->q6
 q2 {(0,1) (1,1) (2,1)} a->q7 b->q8 other->∅
 q3 {(0,1) (1,1)} a->q9 b->q8 other->∅
 q4 {(1,1) (2,1) (3,1)} a->q10 b->q8 other->∅
 q5 {(1,1) (2,0) (3,1)} a->q11 b->q12 other->q12
 q6 {(1,1) (2,1)} a->q13 b->q8 other->∅
 q7 {(1,1) (3,1)} a->q14 b->q8 other->∅
 q8 {(2,1)} a->q13 b->∅ other->∅
 q9 {(1,1)} a->∅ b->q8 other->∅
 q10 {(3,1) (4,1)} a->q14 b->q15 other->∅
 q11 {(2,1) (3,0) (4,1)} a->q16 b->q17 other->q10
 q12 {(2,1) (3,1)} a->q10 b->∅ other->∅
 q13 {(3,1)} a->q14 b->∅ other->∅
 q14 {(4,1)} a->∅ b->q15 other->∅
 q15 {(5,1)} a->q18 b->∅ other->∅
 q16 {(3,1) (4,0) (5,1)} a->q19 b->q20 other->q21
 q17 {(3,1) (4,1) (5,1)} a->q22 b->q15 other->∅
*q18 {(6,1)} 距離1 a->∅ b->∅ other->∅
*q19 {(4,1) (5,1) (6,1)} 距離1 a->q18 b->q15 other->∅
*q20 {(4,1) (5,0) (6,1)} 距離1 a->q23 b->q24 other->q24
 q21 {(4,1) (5,1)} a->q18 b->q15 other->∅
*q22 {(4,1) (6,1)} 距離1 a->∅ b->q15 other->∅
*q23 {(5,1) (6,0)} 距離0 a->q18 b->q18 other->q18
*q24 {(5,1) (6,1)} 距離1 a->q18 b->∅ other->∅
`
	if got := d.draw(); got != want {
		t.Errorf("draw() =\n%s\nwant\n%s", got, want)
	}
	if d.States() != 25 || d.Transitions() != 47 || d.Accepting() != 6 {
		t.Errorf("states/transitions/accepting = %d/%d/%d, want 25/47/6", d.States(), d.Transitions(), d.Accepting())
	}
}

// TestDFAKeepsMinimumEditPerPosition は、状態の集合が位置ごとに編集の最小の 1 つだけを持つことを確かめる。
func TestDFAKeepsMinimumEditPerPosition(t *testing.T) {
	t.Parallel()
	for _, q := range []string{"abaaba", "aaaaaaaa", "abcabcab", "カレーライス"} {
		for _, k := range []int{1, 2} {
			d := NewDFA(q, k)
			for s, set := range d.sets {
				for j := 1; j < len(set); j++ {
					if set[j].i <= set[j-1].i {
						t.Errorf("%s k=%d q%d %v: 同じ位置が 2 度ある", q, k, s, set)
					}
				}
			}
		}
	}
}
