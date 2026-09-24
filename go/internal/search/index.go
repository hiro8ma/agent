// Package search は転置インデックスで候補を絞り（マッチング）、候補だけを BM25 で採点する（ランキング）全文検索。
package search

import (
	"cmp"
	"context"
	"math"
	"slices"
	"time"
)

const (
	DefaultK1 = 1.2
	DefaultB  = 0.75
)

// Doc は検索の対象の 1 文書。タイトルと本文は項目ごとに別々に索引する。
type Doc struct {
	Title   string
	Content string
}

type Field int

const (
	FieldTitle Field = iota
	FieldContent
	numFields
)

func (d Doc) field(f Field) string {
	if f == FieldTitle {
		return d.Title
	}
	return d.Content
}

// posting の pos はタイトルでの位置を split 個並べ、続けて本文での位置を並べる。項目ごとに昇順。
type posting struct {
	doc   int32
	split int32
	pos   []int32
}

func (p posting) positions(f Field) []int32 {
	if f == FieldTitle {
		return p.pos[:p.split]
	}
	return p.pos[p.split:]
}

func (p posting) tf(f Field) int { return len(p.positions(f)) }

type Index struct {
	k1, b    float64
	analyzer *Analyzer
	weights  *[numFields]float64
	docs     []Doc
	lengths  [][numFields]int
	avgLen   [numFields]float64
	vocab    map[string]int32
	postings [][]posting
}

type Option func(*Index)

func WithK1(k1 float64) Option { return func(ix *Index) { ix.k1 = k1 } }

func WithB(b float64) Option { return func(ix *Index) { ix.b = b } }

func WithAnalyzer(a *Analyzer) Option {
	return func(ix *Index) {
		if a != nil {
			ix.analyzer = a
		}
	}
}

// WithFieldWeights は項目ごとに長さを正規化してから重みを掛けて足す（BM25F）。指定しなければ項目をまとめて 1 つの文書として BM25 で採点する。
func WithFieldWeights(title, content float64) Option {
	return func(ix *Index) { ix.weights = &[numFields]float64{FieldTitle: title, FieldContent: content} }
}

// New は文書を 1 回だけ走査して転置インデックスを作る。postings は文書 ID の昇順に並ぶ。
func New(docs []Doc, opts ...Option) *Index {
	ix := &Index{
		k1:       DefaultK1,
		b:        DefaultB,
		analyzer: NewAnalyzer(),
		docs:     docs,
		lengths:  make([][numFields]int, len(docs)),
		vocab:    make(map[string]int32),
	}
	for _, o := range opts {
		o(ix)
	}
	var total [numFields]int
	for id, d := range docs {
		ix.add(int32(id), d, &total)
	}
	if len(docs) > 0 {
		for f := range numFields {
			ix.avgLen[f] = float64(total[f]) / float64(len(docs))
		}
	}
	return ix
}

// add は 1 文書の位置を 1 本の配列にまとめて確保し、語ごとの postings はその部分列を指す。
func (ix *Index) termID(term string) int32 {
	if id, ok := ix.vocab[term]; ok {
		return id
	}
	id := int32(len(ix.postings))
	ix.vocab[term] = id
	ix.postings = append(ix.postings, nil)
	return id
}

// postingsOf は語彙に無い語（ID が負）なら空を返す。
func (ix *Index) postingsOf(term int32) []posting {
	if term < 0 {
		return nil
	}
	return ix.postings[term]
}

func (ix *Index) add(id int32, d Doc, total *[numFields]int) {
	type entry struct {
		term int32
		n    [numFields]int32
		next int32
	}
	var (
		tokens  [numFields][]token
		entries []entry
		size    int
	)
	slot := make(map[int32]int)
	ids := make([][]int32, numFields)
	for f := range numFields {
		tokens[f] = ix.analyzer.analyze(d.field(f))
		ix.lengths[id][f] = len(tokens[f])
		total[f] += len(tokens[f])
		size += len(tokens[f])
		ids[f] = make([]int32, len(tokens[f]))
		for j, t := range tokens[f] {
			tid := ix.termID(t.term)
			ids[f][j] = tid
			i, ok := slot[tid]
			if !ok {
				i = len(entries)
				slot[tid] = i
				entries = append(entries, entry{term: tid})
			}
			entries[i].n[f]++
		}
	}
	arena := make([]int32, size)
	start := make([]int32, len(entries))
	var off int32
	for i, e := range entries {
		start[i] = off
		entries[i].next = off
		off += e.n[FieldTitle] + e.n[FieldContent]
	}
	for f := range numFields {
		for j, t := range tokens[f] {
			i := slot[ids[f][j]]
			arena[entries[i].next] = t.pos
			entries[i].next++
		}
	}
	for i, e := range entries {
		ix.postings[e.term] = append(ix.postings[e.term], posting{
			doc:   id,
			split: e.n[FieldTitle],
			pos:   arena[start[i]:e.next:e.next],
		})
	}
}

type Stats struct {
	Docs      int
	Terms     int
	Postings  int
	Positions int
}

func (ix *Index) Stats() Stats {
	s := Stats{Docs: len(ix.docs), Terms: len(ix.vocab)}
	for _, ps := range ix.postings {
		s.Postings += len(ps)
		for _, p := range ps {
			s.Positions += len(p.pos)
		}
	}
	return s
}

type Hit struct {
	Doc   Doc
	Score float64
}

// Result の Scored はマッチングで残った候補の数。MatchTime は候補を絞るまで、RankTime は採点と並べ替えにかかった時間。
type Result struct {
	Hits      []Hit
	Scored    int
	MatchTime time.Duration
	RankTime  time.Duration
}

// Query の Fields を空にするとすべての項目を検索する。Phrase はクエリの索引語が同じ項目で同じ間隔で並ぶ文書だけを残す。
// Synonyms は検索のときだけクエリを広げる辞書で、見出しと置き換え先のどちらかを含むクエリを両方の書き方の OR にする。
type Query struct {
	Text     string
	Fields   []Field
	Phrase   bool
	Synonyms map[string]string
}

func (ix *Index) Search(ctx context.Context, query string, limit int) ([]Doc, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	res := ix.Rank(query, limit)
	docs := make([]Doc, len(res.Hits))
	for i, h := range res.Hits {
		docs[i] = h.Doc
	}
	return docs, nil
}

func (ix *Index) Rank(query string, limit int) Result {
	return ix.RankQuery(Query{Text: query}, limit)
}

func (ix *Index) RankQuery(q Query, limit int) Result {
	start := time.Now()
	sel := selectFields(q.Fields)
	pq := ix.parse(q)
	df := ix.docFreq(pq.terms, sel)
	var (
		candidates []int
		masks      map[int]uint64
	)
	if q.Phrase {
		candidates, masks = ix.matchPhrase(pq, sel)
	} else {
		candidates = ix.match(pq.terms, sel)
	}
	matched := time.Now()
	contrib := make([]float64, len(pq.terms))
	hits := make([]Hit, 0, len(candidates))
	for _, id := range candidates {
		mask := ^uint64(0)
		if masks != nil {
			mask = masks[id]
		}
		hits = append(hits, Hit{Doc: ix.docs[id], Score: ix.score(id, pq, df, sel, mask, contrib)})
	}
	slices.SortStableFunc(hits, func(a, b Hit) int { return cmp.Compare(b.Score, a.Score) })
	if limit >= 0 && len(hits) > limit {
		hits = hits[:limit]
	}
	return Result{Hits: hits, Scored: len(candidates), MatchTime: matched.Sub(start), RankTime: time.Since(matched)}
}

type fieldSet [numFields]bool

func selectFields(fields []Field) fieldSet {
	var s fieldSet
	if len(fields) == 0 {
		for f := range numFields {
			s[f] = true
		}
		return s
	}
	for _, f := range fields {
		if f >= 0 && f < numFields {
			s[f] = true
		}
	}
	return s
}

func (s fieldSet) tf(p posting) int {
	n := 0
	for f := range numFields {
		if s[f] {
			n += p.tf(f)
		}
	}
	return n
}

func (ix *Index) docFreq(terms []int32, sel fieldSet) []int {
	df := make([]int, len(terms))
	all := sel == selectFields(nil)
	for i, t := range terms {
		if all {
			df[i] = len(ix.postingsOf(t))
			continue
		}
		for _, p := range ix.postingsOf(t) {
			if sel.tf(p) > 0 {
				df[i]++
			}
		}
	}
	return df
}

// match はクエリ語の postings の和集合を返す。どの語も含まない文書はここで落ち、採点されない。
func (ix *Index) match(terms []int32, sel fieldSet) []int {
	seen := make(map[int]struct{})
	var ids []int
	for _, t := range terms {
		for _, p := range ix.postingsOf(t) {
			if sel.tf(p) == 0 {
				continue
			}
			if _, ok := seen[int(p.doc)]; !ok {
				seen[int(p.doc)] = struct{}{}
				ids = append(ids, int(p.doc))
			}
		}
	}
	slices.Sort(ids)
	return ids
}

// matchPhrase は書き方ごとに、すべての語を含む文書を最も短い postings から絞り、位置が並ぶかを確かめる。
// 返す mask は文書ごとにフレーズが見つかった書き方の集合。
func (ix *Index) matchPhrase(pq parsedQuery, sel fieldSet) ([]int, map[int]uint64) {
	masks := make(map[int]uint64)
	for vi, v := range pq.variants {
		if len(v.seq) == 0 {
			continue
		}
		lists := make([][]posting, len(v.terms))
		for i, ti := range v.terms {
			lists[i] = ix.postingsOf(pq.terms[ti])
		}
		rarest := slices.MinFunc(lists, func(a, b []posting) int { return cmp.Compare(len(a), len(b)) })
		for _, p := range rarest {
			if ix.phraseIn(int(p.doc), pq, v, sel) {
				masks[int(p.doc)] |= 1 << vi
			}
		}
	}
	ids := make([]int, 0, len(masks))
	for id := range masks {
		ids = append(ids, id)
	}
	slices.Sort(ids)
	return ids, masks
}

func (ix *Index) phraseIn(id int, pq parsedQuery, v variant, sel fieldSet) bool {
	ps := make([]posting, len(v.seq))
	for i, s := range v.seq {
		p, ok := ix.lookup(pq.terms[s.term], id)
		if !ok {
			return false
		}
		ps[i] = p
	}
	for f := range numFields {
		if !sel[f] {
			continue
		}
		for _, base := range ps[0].positions(f) {
			ok := true
			for i := 1; i < len(ps) && ok; i++ {
				_, ok = slices.BinarySearch(ps[i].positions(f), base+v.seq[i].off)
			}
			if ok {
				return true
			}
		}
	}
	return false
}

func (ix *Index) lookup(term int32, id int) (posting, bool) {
	ps := ix.postingsOf(term)
	i, ok := slices.BinarySearchFunc(ps, int32(id), func(p posting, id int32) int { return cmp.Compare(p.doc, id) })
	if !ok {
		return posting{}, false
	}
	return ps[i], true
}

// score は書き方ごとに BM25 を求め、最も高いものを返す。同じ意味の語を二重に数えないため和ではなく最大を取る。
func (ix *Index) score(id int, pq parsedQuery, df []int, sel fieldSet, mask uint64, contrib []float64) float64 {
	for i, t := range pq.terms {
		contrib[i] = 0
		if df[i] == 0 {
			continue
		}
		p, ok := ix.lookup(t, id)
		if !ok {
			continue
		}
		tf := ix.normalizedTF(id, p, sel)
		if tf == 0 {
			continue
		}
		contrib[i] = ix.idf(df[i]) * tf * (ix.k1 + 1) / (tf + ix.k1)
	}
	best := 0.0
	for vi, v := range pq.variants {
		if mask&(1<<vi) == 0 {
			continue
		}
		s := 0.0
		for _, ti := range v.terms {
			s += contrib[ti]
		}
		best = max(best, s)
	}
	return best
}

// normalizedTF は出現回数を文書の長さで割る。tf/norm で置けば BM25 の tf*(k1+1)/(tf+k1*norm) と同じ値になる。
func (ix *Index) normalizedTF(id int, p posting, sel fieldSet) float64 {
	if ix.weights == nil {
		var tf, length int
		var avg float64
		for f := range numFields {
			if sel[f] {
				tf += p.tf(f)
				length += ix.lengths[id][f]
				avg += ix.avgLen[f]
			}
		}
		return float64(tf) / (1 - ix.b + ix.b*float64(length)/avg)
	}
	s := 0.0
	for f := range numFields {
		tf := p.tf(f)
		if !sel[f] || tf == 0 {
			continue
		}
		norm := 1 - ix.b + ix.b*float64(ix.lengths[id][f])/ix.avgLen[f]
		s += ix.weights[f] * float64(tf) / norm
	}
	return s
}

func (ix *Index) idf(df int) float64 {
	n := float64(len(ix.docs))
	return math.Log(1 + (n-float64(df)+0.5)/(float64(df)+0.5))
}
