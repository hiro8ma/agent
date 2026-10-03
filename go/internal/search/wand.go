package search

import (
	"math"
	"slices"
	"sync"
	"time"
)

// Pruning は OperatorOr の上位 K 件を求めるときに、上位に入りえない文書の採点を省く方法。結果は PruningNone と同じになる。
// 既定（ゼロ値）の PruningWAND は語ごとの点数の上限の和がしきい値（K 件目の点数）を超える文書だけを採点する。PruningNone は全候補を採点する。
// PruningBlockMaxWAND は postings を BlockSize 件ごとの区間に分け、区間ごとの上限でさらに区間をまとめて飛ばす。
// 使えるのは RankingBM25 と RankingTFIDF で、書き方が 1 つ、各語の一致させる索引語が 1 つ（Synonyms / Typo / Prefix / Thesaurus / Extra を使わない）、limit が正のときだけ。
// それ以外は PruningNone と同じく全候補を採点する。Result.Scored は実際に採点した文書の数になる。
type Pruning int

const (
	PruningWAND Pruning = iota
	PruningNone
	PruningBlockMaxWAND
)

const DefaultBlockSize = 64

// boundSlack は上限に掛ける余裕。点数の和を足す順序が採点と違い、丸めで上限を超えないようにする。
const boundSlack = 1 + 1e-9

type boundKind uint8

const (
	boundBM25 boundKind = iota
	boundTFIDF
)

// boundKey の avg と b は BM25 の長さの正規化に効く。TF-IDF では使わないので 0 にする。block が 0 なら postings 全体を 1 区間にする。
type boundKey struct {
	term  int32
	sel   int
	block int
	kind  boundKind
	b     float64
	avg   [numFields]float64
}

// boundCache は語と区間ごとの、正規化した出現回数の最大を持つ。IDF を掛ける前の値なので、文書数と文書頻度がクエリごとに変わっても使える。
// other は索引の外の統計（DFS と Updatable）の平均文書長で、別の値が来たら前の値の分を捨てる。
type boundCache struct {
	mu    sync.Mutex
	m     map[boundKey][]float64
	other [numFields]float64
}

func newBoundCache() *boundCache { return &boundCache{m: make(map[boundKey][]float64)} }

// maxima は語の postings を block 件ごとに区切り、区間ごとに採点と同じ式で求めた出現回数の最大を返す。
func (ix *Index) maxima(term int32, sel fieldSet, block int, kind boundKind, avg [numFields]float64) []float64 {
	k := boundKey{term: term, sel: sel.bits(), block: block, kind: kind}
	if kind == boundBM25 {
		k.b, k.avg = ix.b, avg
	}
	bc := ix.bounds
	bc.mu.Lock()
	defer bc.mu.Unlock()
	if m, ok := bc.m[k]; ok {
		return m
	}
	if kind == boundBM25 && avg != ix.avgLen && avg != bc.other {
		for key := range bc.m {
			if key.kind == boundBM25 && key.avg == bc.other {
				delete(bc.m, key)
			}
		}
		bc.other = avg
	}
	ps := ix.postingsOf(term)
	size := block
	if size <= 0 {
		size = max(len(ps), 1)
	}
	all := sel == selectFields(nil)
	m := make([]float64, (len(ps)+size-1)/size)
	for i, p := range ps {
		if !all && sel.tf(p) == 0 {
			continue
		}
		var v float64
		if kind == boundBM25 {
			v = ix.normalizedTF(term, int(p.doc), p, sel, avg)
		} else {
			v = ix.effectiveTF(term, p, sel)
		}
		m[i/size] = max(m[i/size], v)
	}
	bc.m[k] = m
	return m
}

type wandCursor struct {
	term   int
	ps     []posting
	pos    int
	ub     float64
	maxima []float64
	sb     int
	// weight は語の IDF。正規化した出現回数の最大から点数の上限を求めるときにも掛ける。
	weight float64
}

func (c *wandCursor) done() bool  { return c.pos >= len(c.ps) }
func (c *wandCursor) doc() int32  { return c.ps[c.pos].doc }
func (c *wandCursor) last() int32 { return c.ps[len(c.ps)-1].doc }

type wandRanker struct {
	ix     *Index
	pq     parsedQuery
	sel    fieldSet
	all    bool
	avg    [numFields]float64
	kind   boundKind
	tf     TFWeight
	block  int
	cs     []*wandCursor
	scored int
}

// eligibleForPruning は採点の式が語ごとの点数の和で、文書ごとの割り算や最大を取る処理を含まないクエリかを返す。
func eligibleForPruning(q Query, pq parsedQuery, limit int) bool {
	if q.Pruning == PruningNone || limit <= 0 || q.Expr != nil || q.Phrase || q.Near > 0 || q.Operator != OperatorOr {
		return false
	}
	if q.Ranking != RankingBM25 && q.Ranking != RankingTFIDF {
		return false
	}
	if len(pq.variants) != 1 || pq.variants[0].grouped != nil || len(pq.extra) > 0 {
		return false
	}
	seen := make(map[int]bool)
	for _, slot := range pq.variants[0].slots {
		if len(slot) != 1 || seen[slot[0].term] {
			return false
		}
		seen[slot[0].term] = true
	}
	return true
}

// rankPruned は語ごとのカーソルを文書番号の順に進め、上限の和がしきい値を超える文書（ピボット）だけを採点する。
// 採点の式とヒープは rankWith と同じものを使い、同点は文書番号の小さい方を上にする。
func (ix *Index) rankPruned(q Query, pq parsedQuery, df []int, sel fieldSet, limit int, c *corpus, start time.Time) (Result, []int) {
	r := &wandRanker{ix: ix, pq: pq, sel: sel, all: sel == selectFields(nil), avg: c.avgLen, tf: q.TFIDF.TF}
	if q.Pruning == PruningBlockMaxWAND {
		r.block = q.BlockSize
		if r.block <= 0 {
			r.block = DefaultBlockSize
		}
	}
	var idfs []float64
	if q.Ranking == RankingTFIDF {
		r.kind = boundTFIDF
		idfs = tfidfWeights(q.TFIDF, pq, df, c)
	}
	for i, t := range pq.terms {
		ps := ix.postingsOf(t)
		if len(ps) == 0 {
			continue
		}
		var w float64
		switch {
		case r.kind == boundTFIDF:
			w = idfs[i]
		case df[i] > 0:
			w = ix.idf(c.docs, df[i])
		}
		cur := &wandCursor{term: i, ps: ps, weight: w}
		r.skipUnselected(cur)
		if cur.done() {
			continue
		}
		cur.ub = r.bound(w, slices.Max(ix.maxima(t, sel, 0, r.kind, r.avg)))
		if r.block > 0 {
			cur.maxima = ix.maxima(t, sel, r.block, r.kind, r.avg)
		}
		r.cs = append(r.cs, cur)
	}
	matched := time.Now()
	top := newRankTopWith(0, limit, true)
	r.run(top, limit, idfs)
	ranked := top.result()
	hits := make([]Hit, len(ranked))
	ids := make([]int, len(ranked))
	for i, s := range ranked {
		hits[i] = Hit{Doc: ix.docs[s.ord], Score: s.score}
		ids[i] = s.ord
	}
	return Result{
		Hits: hits, Scored: r.scored, Expanded: pq.expanded, ExpansionDropped: pq.dropped,
		MatchTime: matched.Sub(start), RankTime: time.Since(matched),
	}, ids
}

func (r *wandRanker) bound(weight, m float64) float64 { return r.exactBound(weight, m) * boundSlack }

// exactBound は正規化した出現回数の最大 m から、採点と同じ式で語の点数の最大を求める。BM25 の m(k1+1)/(m+k1) と TF の重みは m について単調に増える。
func (r *wandRanker) exactBound(weight, m float64) float64 {
	var s float64
	if r.kind == boundBM25 {
		if m == 0 {
			return 0
		}
		s = weight * m * (r.ix.k1 + 1) / (m + r.ix.k1)
	} else {
		s = r.tf.weight(m) * weight
	}
	return max(s, 0)
}

func (r *wandRanker) skipUnselected(c *wandCursor) {
	if r.all {
		return
	}
	for c.pos < len(c.ps) && r.sel.tf(c.ps[c.pos]) == 0 {
		c.pos++
	}
}

// advance はカーソルを文書番号が target 以上の位置まで、間隔を倍にして越えてから二分探索で進める。
func (r *wandRanker) advance(c *wandCursor, target int32) {
	ps := c.ps
	if c.pos >= len(ps) || ps[c.pos].doc >= target {
		return
	}
	lo, step := c.pos, 1
	for lo+step < len(ps) && ps[lo+step].doc < target {
		lo += step
		step *= 2
	}
	hi := min(lo+step+1, len(ps))
	k, _ := slices.BinarySearchFunc(ps[lo:hi], target, func(p posting, id int32) int { return int(p.doc - id) })
	c.pos = lo + k
	r.skipUnselected(c)
}

func (r *wandRanker) next(c *wandCursor) {
	c.pos++
	r.skipUnselected(c)
}

// blockAt は doc 以上の最初の出現記録を含む区間の上限と、その区間の最後の文書番号を返す。そのような出現記録が無ければ上限 0 で ok は false。
// ピボットの文書番号は減らないので、区間の位置は前回から前にだけ進める。
func (r *wandRanker) blockAt(c *wandCursor, doc int32) (float64, int32, bool) {
	n := len(c.maxima)
	c.sb = max(c.sb, c.pos/r.block)
	for c.sb < n && c.ps[min((c.sb+1)*r.block, len(c.ps))-1].doc < doc {
		c.sb++
	}
	if c.sb >= n {
		return 0, 0, false
	}
	return r.bound(c.weight, c.maxima[c.sb]), c.ps[min((c.sb+1)*r.block, len(c.ps))-1].doc, true
}

func (r *wandRanker) run(top *rankTop, limit int, idfs []float64) {
	contrib := make([]float64, len(r.pq.terms))
	theta := math.Inf(-1)
	cs := r.cs
	for {
		cs = slices.DeleteFunc(cs, (*wandCursor).done)
		if len(cs) == 0 {
			return
		}
		slices.SortFunc(cs, func(a, b *wandCursor) int { return int(a.doc() - b.doc()) })
		p, acc := -1, 0.0
		for i, c := range cs {
			acc += c.ub
			if acc > theta {
				p = i
				break
			}
		}
		if p < 0 {
			return
		}
		pivot := cs[p].doc()
		for p+1 < len(cs) && cs[p+1].doc() == pivot {
			p++
		}
		if r.block > 0 {
			sum, end := 0.0, int64(math.MaxInt32)
			for _, c := range cs[:p+1] {
				ub, last, ok := r.blockAt(c, pivot)
				if ok {
					sum += ub
					end = min(end, int64(last))
				}
			}
			if sum <= theta {
				// pivot から区間の終わりまでの文書は、ピボットより前の語の区間にしか現れず、その上限の和がしきい値以下。
				next := end + 1
				if p+1 < len(cs) {
					next = min(next, int64(cs[p+1].doc()))
				}
				for _, c := range cs[:p+1] {
					if next > int64(c.last()) {
						c.pos = len(c.ps)
						continue
					}
					r.advance(c, int32(next))
				}
				continue
			}
		}
		if cs[0].doc() != pivot {
			for _, c := range cs[:p+1] {
				r.advance(c, pivot)
			}
			continue
		}
		clear(contrib)
		for _, c := range cs[:p+1] {
			contrib[c.term] = r.contribution(c, idfs)
		}
		r.scored++
		top.push(scored{ord: int(pivot), score: combine(r.pq, ^uint64(0), contrib)})
		if len(top.items) == limit {
			theta = top.items[0].score
		}
		for _, c := range cs[:p+1] {
			r.next(c)
		}
	}
}

// contribution は score / tfidfScore と同じ式と同じ演算の順で、カーソルの位置の出現記録から語の点数を求める。
func (r *wandRanker) contribution(c *wandCursor, idfs []float64) float64 {
	ix, p := r.ix, c.ps[c.pos]
	t := r.pq.terms[c.term]
	if r.kind == boundTFIDF {
		if idfs[c.term] == 0 {
			return 0
		}
		return r.tf.weight(ix.effectiveTF(t, p, r.sel)) * idfs[c.term]
	}
	if c.weight == 0 {
		return 0
	}
	tf := ix.normalizedTF(t, int(p.doc), p, r.sel, r.avg)
	if tf == 0 {
		return 0
	}
	return c.weight * tf * (ix.k1 + 1) / (tf + ix.k1)
}
