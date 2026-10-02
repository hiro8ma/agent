package fst

import (
	"fmt"
	"strings"
	"testing"
)

// draw は FST の辺を「状態 -文字/出力-> 状態」の形で、根から幅優先に並べる。状態の番号は訪ねた順に振り直す。
// 複数の辺が同じ状態に入っていれば、そこで経路が合流している（トライ木なら別々の枝になる）。
func (f *FST) draw() string {
	names := map[int]int{f.root: 0}
	queue := []int{f.root}
	var b strings.Builder
	for len(queue) > 0 {
		s := queue[0]
		queue = queue[1:]
		n := &f.nodes[s]
		if n.final {
			fmt.Fprintf(&b, "s%d 終了 出力 %d\n", names[s], n.finalOut)
		}
		for _, t := range n.trans {
			if _, ok := names[t.to]; !ok {
				names[t.to] = len(names)
				queue = append(queue, t.to)
			}
			fmt.Fprintf(&b, "s%d -%c/%d-> s%d\n", names[s], t.label, t.out, names[t.to])
		}
	}
	return b.String()
}

func TestDrawSharedSuffix(t *testing.T) {
	t.Parallel()
	f, err := Build([]string{"カツカレー", "キーマカレー"}, []uint64{0, 7})
	if err != nil {
		t.Fatal(err)
	}
	// カツ と キーマ の後ろは同じ s3 に入り、カレー の 3 文字を 1 本で共有する。
	// 値の違い（0 と 7）は最初の辺（キ/7）に乗り、残りの辺の出力は 0 なので、合流しても値が混ざらない。
	want := `s0 -カ/0-> s1
s0 -キ/7-> s2
s1 -ツ/0-> s3
s2 -ー/0-> s4
s3 -カ/0-> s5
s4 -マ/0-> s3
s5 -レ/0-> s6
s6 -ー/0-> s7
s7 終了 出力 0
`
	if got := f.draw(); got != want {
		t.Errorf("draw() =\n%s\nwant\n%s", got, want)
	}
}
