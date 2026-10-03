package ann

import "slices"

// Neighbor の Dist は L2 距離の 2 乗。近似の索引では量子化した値なので、全探索の値とは一致しない。
type Neighbor struct {
	ID   int32
	Dist float32
}

// closer は a が b より近いか。同じ距離なら ID の小さい方を近いとみなし、結果を決定的にする。
func closer(a, b Neighbor) bool {
	return a.Dist < b.Dist || (a.Dist == b.Dist && a.ID < b.ID)
}

func compareNeighbor(a, b Neighbor) int {
	switch {
	case closer(a, b):
		return -1
	case closer(b, a):
		return 1
	}
	return 0
}

// kBest は近い順に k 件を持つ最大ヒープ。根が k 件の中で最も遠い候補になる。
type kBest struct {
	k int
	h []Neighbor
}

func newKBest(k int) *kBest { return &kBest{k: k, h: make([]Neighbor, 0, k)} }

func (t *kBest) full() bool { return len(t.h) >= t.k }

// worst は k 件そろっていなければ +Inf 相当として扱えるよう ok=false を返す。
func (t *kBest) worst() (Neighbor, bool) {
	if !t.full() || t.k == 0 {
		return Neighbor{}, false
	}
	return t.h[0], true
}

func (t *kBest) push(n Neighbor) bool {
	if len(t.h) < t.k {
		t.h = append(t.h, n)
		t.up(len(t.h) - 1)
		return true
	}
	if t.k == 0 || !closer(n, t.h[0]) {
		return false
	}
	t.h[0] = n
	t.down(0)
	return true
}

func (t *kBest) up(i int) {
	for i > 0 {
		p := (i - 1) / 2
		if !closer(t.h[p], t.h[i]) {
			return
		}
		t.h[i], t.h[p] = t.h[p], t.h[i]
		i = p
	}
}

func (t *kBest) down(i int) {
	n := len(t.h)
	for {
		l, s := 2*i+1, i
		if l < n && closer(t.h[s], t.h[l]) {
			s = l
		}
		if r := l + 1; r < n && closer(t.h[s], t.h[r]) {
			s = r
		}
		if s == i {
			return
		}
		t.h[i], t.h[s] = t.h[s], t.h[i]
		i = s
	}
}

func (t *kBest) sorted() []Neighbor {
	out := slices.Clone(t.h)
	slices.SortFunc(out, compareNeighbor)
	return out
}

// minQueue は近い順に取り出す最小ヒープ。HNSW の探索で次に広げる候補を持つ。
type minQueue []Neighbor

func (q *minQueue) push(n Neighbor) {
	*q = append(*q, n)
	h := *q
	for i := len(h) - 1; i > 0; {
		p := (i - 1) / 2
		if !closer(h[i], h[p]) {
			break
		}
		h[i], h[p] = h[p], h[i]
		i = p
	}
}

func (q *minQueue) pop() Neighbor {
	h := *q
	top := h[0]
	last := len(h) - 1
	h[0] = h[last]
	h = h[:last]
	for i := 0; ; {
		l, s := 2*i+1, i
		if l < len(h) && closer(h[l], h[s]) {
			s = l
		}
		if r := l + 1; r < len(h) && closer(h[r], h[s]) {
			s = r
		}
		if s == i {
			break
		}
		h[i], h[s] = h[s], h[i]
		i = s
	}
	*q = h
	return top
}

// selectK は ns を並べ替え、先頭の k 件を近い k 件にする。先頭 k 件の中の順序は決めない（NumPy の argpartition と同じ）。
func selectK(ns []Neighbor, k int) {
	if k <= 0 || k >= len(ns) {
		return
	}
	lo, hi := 0, len(ns)-1
	for lo < hi {
		p := partition(ns, lo, hi)
		switch {
		case p == k-1:
			return
		case p < k-1:
			lo = p + 1
		default:
			hi = p - 1
		}
	}
}

// partition は中央値の 3 点を軸に Lomuto 分割し、軸の最終位置を返す。
func partition(ns []Neighbor, lo, hi int) int {
	mid := lo + (hi-lo)/2
	if closer(ns[mid], ns[lo]) {
		ns[mid], ns[lo] = ns[lo], ns[mid]
	}
	if closer(ns[hi], ns[lo]) {
		ns[hi], ns[lo] = ns[lo], ns[hi]
	}
	if closer(ns[mid], ns[hi]) {
		ns[mid], ns[hi] = ns[hi], ns[mid]
	}
	pivot := ns[hi]
	i := lo
	for j := lo; j < hi; j++ {
		if closer(ns[j], pivot) {
			ns[i], ns[j] = ns[j], ns[i]
			i++
		}
	}
	ns[i], ns[hi] = ns[hi], ns[i]
	return i
}

// Recall は want の ID のうち got に含まれた割合。
func Recall(got, want []Neighbor) float64 {
	if len(want) == 0 {
		return 1
	}
	ids := make(map[int32]struct{}, len(got))
	for _, n := range got {
		ids[n.ID] = struct{}{}
	}
	hit := 0
	for _, n := range want {
		if _, ok := ids[n.ID]; ok {
			hit++
		}
	}
	return float64(hit) / float64(len(want))
}
