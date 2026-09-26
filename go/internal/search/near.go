package search

import "slices"

// nearIn は項目ごとに、slot ごとの位置の列から 1 つずつ選んだ組のうち最初と最後の差が最も小さいものを求め、k 以内かを返す。
func (ix *Index) nearIn(id int, pq parsedQuery, v variant, sel fieldSet, k int) bool {
	for f := range numFields {
		if !sel[f] {
			continue
		}
		lists := make([][]int32, len(v.slots))
		for i, slot := range v.slots {
			for _, p := range ix.present(id, pq, slot) {
				lists[i] = append(lists[i], p.positions(Field(f))...)
			}
			slices.Sort(lists[i])
		}
		if w, ok := minWindow(lists); ok && w <= k {
			return true
		}
	}
	return false
}

// minWindow は昇順の列それぞれから 1 つずつ選んだときの、最大と最小の差の最小値を返す。空の列があれば false。
func minWindow(lists [][]int32) (int, bool) {
	if len(lists) == 0 {
		return 0, false
	}
	for _, l := range lists {
		if len(l) == 0 {
			return 0, false
		}
	}
	idx := make([]int, len(lists))
	best := -1
	for {
		lo, hi := 0, lists[0][idx[0]]
		for i := range lists {
			x := lists[i][idx[i]]
			if x < lists[lo][idx[lo]] {
				lo = i
			}
			hi = max(hi, x)
		}
		if w := int(hi - lists[lo][idx[lo]]); best < 0 || w < best {
			best = w
		}
		idx[lo]++
		if idx[lo] == len(lists[lo]) {
			return best, true
		}
	}
}
