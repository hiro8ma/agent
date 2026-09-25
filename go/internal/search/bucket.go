package search

import (
	"math"
	"slices"
)

// Ranking は候補の並べ方。RankingBM25 は BM25 の点数、RankingBucket は規則を順に比べ、前の規則で差がついた順は後の規則で入れ替えない。
// RankingBucket の規則は Meilisearch の既定から sort と word position を除いた words / typo / proximity / attribute / exactness。
type Ranking int

const (
	RankingBM25 Ranking = iota
	RankingBucket
)

// maxProximity は語の組の距離の上限。同じ項目に無い組もこの値にする。
const maxProximity = 8

// bucketKey の値は大きいほど上位。typos と proximity は小さいほど上位なので、比べる前に符号を逆にする。
type bucketKey struct {
	words     int
	typos     int
	proximity int
	attribute int
	exactness int
}

// encode は規則を上位から桁に詰めた整数を返す。点数の大小で比べるだけで規則の順に並び、シャードをまたいでも同じ順にできる。
func (k bucketKey) encode() float64 {
	digits := []struct{ v, width int }{
		{k.words, 1 << 8},
		{(1<<8 - 1) - k.typos, 1 << 8},
		{(1<<10 - 1) - k.proximity, 1 << 10},
		{k.attribute, 1 << 8},
		{k.exactness, 1 << 8},
	}
	s := 0.0
	for _, d := range digits {
		s = s*float64(d.width) + float64(min(max(d.v, 0), d.width-1))
	}
	return s
}

// bucketScore は書き方ごとに規則の値を求め、最も上位のものを返す。
func (ix *Index) bucketScore(id int, pq parsedQuery, sel fieldSet, mask uint64) float64 {
	best := math.Inf(-1)
	for vi, v := range pq.variants {
		if mask&(1<<vi) == 0 {
			continue
		}
		best = max(best, ix.bucketKey(id, pq, v, sel).encode())
	}
	return best
}

// bucketKey の words は文書に現れたクエリの語の数、typos はその語の打ち間違いの和、proximity はクエリの順に隣り合う語の組ごとの最小の距離の和、
// attribute はタイトルに現れた語の数、exactness は打ち間違いも接頭辞の補完も無く一致した語の数。
func (ix *Index) bucketKey(id int, pq parsedQuery, v variant, sel fieldSet) bucketKey {
	var k bucketKey
	var matched [][numFields][]int32
	for _, slot := range v.slots {
		var (
			found bool
			typos = math.MaxInt
			exact bool
			title bool
			pos   [numFields][]int32
		)
		for _, a := range slot {
			p, ok := ix.lookup(pq.terms[a.term], id)
			if !ok || sel.tf(p) == 0 {
				continue
			}
			found = true
			typos = min(typos, a.typos)
			exact = exact || a.exact
			title = title || (sel[FieldTitle] && p.tf(FieldTitle) > 0)
			for f := range numFields {
				if sel[f] {
					pos[f] = append(pos[f], p.positions(Field(f))...)
				}
			}
		}
		if !found {
			continue
		}
		k.words++
		k.typos += typos
		if exact {
			k.exactness++
		}
		if title {
			k.attribute++
		}
		for f := range pos {
			slices.Sort(pos[f])
		}
		matched = append(matched, pos)
	}
	for i := 1; i < len(matched); i++ {
		d := maxProximity
		for f := range numFields {
			d = min(d, minDistance(matched[i-1][f], matched[i][f]))
		}
		k.proximity += d
	}
	return k
}

// minDistance は昇順の 2 つの位置の列から最も近い組の距離を返す。どちらかが空なら maxProximity。
func minDistance(a, b []int32) int {
	d := maxProximity
	i, j := 0, 0
	for i < len(a) && j < len(b) {
		d = min(d, int(abs32(a[i]-b[j])))
		if a[i] < b[j] {
			i++
		} else {
			j++
		}
	}
	return d
}

func abs32(x int32) int32 {
	if x < 0 {
		return -x
	}
	return x
}
