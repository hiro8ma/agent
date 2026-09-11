// Package index はメモリ上の全探索ベクトル検索。距離関数を差し替えてスカラと SIMD を同じ走査で比べる。
package index

import "slices"

type Result struct {
	ID    int
	Score float32
}

// worse は a が b より順位が下か。同点は ID が大きい方を下にして結果を決定的にする。
func worse(a, b Result) bool {
	if a.Score != b.Score {
		return a.Score < b.Score
	}
	return a.ID > b.ID
}

// topK は上位 k 件を持つ最小ヒープ。先頭が k 件の中で一番下の候補。
type topK struct {
	k int
	h []Result
}

func newTopK(k int) *topK { return &topK{k: k, h: make([]Result, 0, k)} }

func (t *topK) push(id int, score float32) {
	r := Result{ID: id, Score: score}
	if len(t.h) < t.k {
		t.h = append(t.h, r)
		t.up(len(t.h) - 1)
		return
	}
	if t.k == 0 || !worse(t.h[0], r) {
		return
	}
	t.h[0] = r
	t.down(0)
}

func (t *topK) up(i int) {
	for i > 0 {
		p := (i - 1) / 2
		if !worse(t.h[i], t.h[p]) {
			return
		}
		t.h[i], t.h[p] = t.h[p], t.h[i]
		i = p
	}
}

func (t *topK) down(i int) {
	n := len(t.h)
	for {
		l, s := 2*i+1, i
		if l < n && worse(t.h[l], t.h[s]) {
			s = l
		}
		if r := l + 1; r < n && worse(t.h[r], t.h[s]) {
			s = r
		}
		if s == i {
			return
		}
		t.h[i], t.h[s] = t.h[s], t.h[i]
		i = s
	}
}

// results はスコアの高い順に返す。
func (t *topK) results() []Result {
	out := slices.Clone(t.h)
	slices.SortFunc(out, func(a, b Result) int {
		switch {
		case worse(b, a):
			return -1
		case worse(a, b):
			return 1
		}
		return 0
	})
	return out
}

// Recall は want の ID のうち got に含まれた割合。
func Recall(got, want []Result) float64 {
	if len(want) == 0 {
		return 1
	}
	ids := make(map[int]bool, len(got))
	for _, r := range got {
		ids[r.ID] = true
	}
	hit := 0
	for _, r := range want {
		if ids[r.ID] {
			hit++
		}
	}
	return float64(hit) / float64(len(want))
}
