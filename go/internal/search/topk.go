package search

import (
	"cmp"
	"slices"
)

// heapMinRatio は候補の数が limit のこの倍以上のときに、全件を並べ替えずに大きさ limit のヒープで上位を持つ。BenchmarkTopKRatio で比 2 からヒープが速かった。
const heapMinRatio = 2

// scored の ord は候補の中の順番。同点はこの順で並べ、安定ソートと同じ結果にする。
type scored struct {
	ord   int
	score float64
}

// above は a が b より上位かを返す。点数が高い方、同点なら ord が小さい方を上にする。
func above(a, b scored) bool {
	if c := cmp.Compare(a.score, b.score); c != 0 {
		return c > 0
	}
	return a.ord < b.ord
}

func compareRank(a, b scored) int {
	return cmp.Or(cmp.Compare(b.score, a.score), cmp.Compare(a.ord, b.ord))
}

// rankTop は push された候補から上位 limit 件を返す。useHeap なら根に最も下位の候補を置くヒープで limit 件だけを持つ。
type rankTop struct {
	limit   int
	useHeap bool
	items   []scored
}

func newRankTop(candidates, limit int) *rankTop {
	useHeap := limit >= 0 && candidates >= heapMinRatio*limit && candidates > limit
	return newRankTopWith(candidates, limit, useHeap)
}

func newRankTopWith(candidates, limit int, useHeap bool) *rankTop {
	if useHeap {
		return &rankTop{limit: limit, useHeap: true, items: make([]scored, 0, limit)}
	}
	return &rankTop{limit: limit, items: make([]scored, 0, candidates)}
}

// push は ord の昇順に呼ぶ。後から来た同点の候補はヒープの根を追い出さない。
func (t *rankTop) push(s scored) {
	if !t.useHeap {
		t.items = append(t.items, s)
		return
	}
	if t.limit == 0 {
		return
	}
	if len(t.items) < t.limit {
		t.items = append(t.items, s)
		t.up(len(t.items) - 1)
		return
	}
	if above(s, t.items[0]) {
		t.items[0] = s
		t.down(0)
	}
}

func (t *rankTop) up(i int) {
	h := t.items
	for i > 0 {
		p := (i - 1) / 2
		if !above(h[p], h[i]) {
			return
		}
		h[p], h[i] = h[i], h[p]
		i = p
	}
}

func (t *rankTop) down(i int) {
	h := t.items
	for {
		lowest, l, r := i, 2*i+1, 2*i+2
		if l < len(h) && above(h[lowest], h[l]) {
			lowest = l
		}
		if r < len(h) && above(h[lowest], h[r]) {
			lowest = r
		}
		if lowest == i {
			return
		}
		h[i], h[lowest] = h[lowest], h[i]
		i = lowest
	}
}

func (t *rankTop) result() []scored {
	if t.useHeap {
		slices.SortFunc(t.items, compareRank)
		return t.items
	}
	slices.SortFunc(t.items, compareRank)
	if t.limit >= 0 && len(t.items) > t.limit {
		return t.items[:t.limit]
	}
	return t.items
}
