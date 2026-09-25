package search

import (
	"cmp"
	"slices"
)

// Operator はクエリの索引語をどう組み合わせて候補にするかを決める。OperatorOr はどれか 1 つ、OperatorAnd はすべてを含む文書を候補にする。
type Operator int

const (
	OperatorOr Operator = iota
	OperatorAnd
)

// defaultGallopRatio は長い方の postings が短い方の何倍以上なら galloping で探すか。BenchmarkIntersect で比 16 では 2 つのポインタが速く、20 では galloping が速かった。
const defaultGallopRatio = 20

// matchAll は書き方ごとに、すべての語を含む文書を短い postings から順に共通部分を取って絞る。keep が nil でなければ、残った文書のうち keep を満たすものだけを返す。
// 書き方が 1 つなら mask は nil で返し、すべての書き方で採点させる。
func (ix *Index) matchAll(pq parsedQuery, sel fieldSet, keep func(id int, v variant) bool) ([]int, map[int]uint64) {
	all := sel == selectFields(nil)
	perVariant := make([][]int32, len(pq.variants))
	for vi, v := range pq.variants {
		if len(v.slots) == 0 {
			continue
		}
		lists := make([][]posting, len(v.slots))
		for i, slot := range v.slots {
			lists[i] = ix.slotPostings(pq, slot, sel, all)
		}
		slices.SortFunc(lists, func(a, b []posting) int { return cmp.Compare(len(a), len(b)) })
		ids := docsOf(lists[0], sel, all)
		for _, ps := range lists[1:] {
			if len(ids) == 0 {
				break
			}
			ids = intersect(ids, ps, sel, all, ix.gallopRatio)
		}
		if keep != nil {
			ids = slices.DeleteFunc(ids, func(id int32) bool { return !keep(int(id), v) })
		}
		perVariant[vi] = ids
	}
	if len(pq.variants) == 1 {
		out := make([]int, len(perVariant[0]))
		for i, id := range perVariant[0] {
			out[i] = int(id)
		}
		return out, nil
	}
	masks := make(map[int]uint64)
	for vi, ids := range perVariant {
		for _, id := range ids {
			masks[int(id)] |= 1 << vi
		}
	}
	out := make([]int, 0, len(masks))
	for id := range masks {
		out = append(out, id)
	}
	slices.Sort(out)
	return out, masks
}

// slotPostings は slot の索引語の postings を文書番号で合わせ、文書ごとに選んだ項目に現れる 1 つを残す。索引語が 1 つならそのまま返す。
func (ix *Index) slotPostings(pq parsedQuery, slot []alt, sel fieldSet, all bool) []posting {
	if len(slot) == 1 {
		return ix.postingsOf(pq.terms[slot[0].term])
	}
	var out []posting
	for _, a := range slot {
		for _, p := range ix.postingsOf(pq.terms[a.term]) {
			if all || sel.tf(p) > 0 {
				out = append(out, p)
			}
		}
	}
	slices.SortStableFunc(out, func(a, b posting) int { return cmp.Compare(a.doc, b.doc) })
	return slices.CompactFunc(out, func(a, b posting) bool { return a.doc == b.doc })
}

func docsOf(ps []posting, sel fieldSet, all bool) []int32 {
	ids := make([]int32, 0, len(ps))
	for _, p := range ps {
		if all || sel.tf(p) > 0 {
			ids = append(ids, p.doc)
		}
	}
	return ids
}

// intersect は昇順の ids と postings の共通部分を ids の領域に書いて返す。ratio が 0 以下なら常に 2 つのポインタで進める。
func intersect(ids []int32, ps []posting, sel fieldSet, all bool, ratio int) []int32 {
	if useGallop(len(ids), len(ps), ratio) {
		return intersectGallop(ids, ps, sel, all)
	}
	return intersectMerge(ids, ps, sel, all)
}

func useGallop(short, long, ratio int) bool {
	return ratio > 0 && long >= ratio*short
}

func intersectMerge(ids []int32, ps []posting, sel fieldSet, all bool) []int32 {
	out := ids[:0]
	i, j := 0, 0
	for i < len(ids) && j < len(ps) {
		switch a, b := ids[i], ps[j].doc; {
		case a < b:
			i++
		case a > b:
			j++
		default:
			if all || sel.tf(ps[j]) > 0 {
				out = append(out, a)
			}
			i++
			j++
		}
	}
	return out
}

// intersectGallop は ids の各要素を、ps の前回の位置から 1, 2, 4, ... と間隔を倍にして越えるまで進め、その区間を二分探索する。
func intersectGallop(ids []int32, ps []posting, sel fieldSet, all bool) []int32 {
	out := ids[:0]
	j := 0
	for _, a := range ids {
		if j >= len(ps) {
			break
		}
		lo, step := j, 1
		for lo+step < len(ps) && ps[lo+step].doc < a {
			lo += step
			step *= 2
		}
		hi := min(lo+step+1, len(ps))
		k, found := slices.BinarySearchFunc(ps[lo:hi], a, func(p posting, id int32) int { return cmp.Compare(p.doc, id) })
		j = lo + k
		if found {
			if all || sel.tf(ps[j]) > 0 {
				out = append(out, a)
			}
			j++
		}
	}
	return out
}
