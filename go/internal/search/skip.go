package search

import "math"

// skipPostings は postings に interval 件おきの飛び先を持たせたもの。skips[s] は postings[s*interval] の文書番号で、飛び先の文書番号を本体を読まずに比べるために別の配列に置く。
// interval が 2 未満なら飛び先を持たない。
type skipPostings struct {
	ps       []posting
	interval int
	skips    []int32
}

// sqrtInterval は飛び先の間隔の既定で、長さの平方根（Manning ら『Introduction to Information Retrieval』の √P）。
func sqrtInterval(n int) int { return int(math.Sqrt(float64(n))) }

func newSkipPostings(ps []posting, interval int) *skipPostings {
	sp := &skipPostings{ps: ps, interval: interval}
	if interval < 2 {
		return sp
	}
	sp.skips = make([]int32, 0, (len(ps)+interval-1)/interval)
	for i := 0; i < len(ps); i += interval {
		sp.skips = append(sp.skips, ps[i].doc)
	}
	return sp
}

// intersectSkip は昇順の ids と、飛び先を持つ postings の共通部分を ids の領域に書いて返す。文書番号を比べた回数も返す。
// 飛び先は飛び先を持つ位置（interval の倍数）にいるときだけ見る。飛び先の文書番号が ids の値以下なら、越えるまで飛び先をたどる。
func intersectSkip(ids []int32, sp *skipPostings, sel fieldSet, all bool) ([]int32, int) {
	out := ids[:0]
	ps, iv := sp.ps, sp.interval
	n := 0
	i, j := 0, 0
	for i < len(ids) && j < len(ps) {
		a, b := ids[i], ps[j].doc
		n++
		switch {
		case a == b:
			if all || sel.tf(ps[j]) > 0 {
				out = append(out, a)
			}
			i++
			j++
		case a < b:
			i++
		default:
			if sp.skips == nil || j%iv != 0 {
				j++
				continue
			}
			s := j / iv
			jumped := false
			for s+1 < len(sp.skips) {
				n++
				if sp.skips[s+1] > a {
					break
				}
				s++
				jumped = true
			}
			if jumped {
				j = s * iv
			} else {
				j++
			}
		}
	}
	return out, n
}
