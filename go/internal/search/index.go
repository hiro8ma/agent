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

// Doc は検索の対象の 1 文書。タイトルと本文は項目ごとに別々に索引する。ID は索引せず、検索結果を呼び出し側の文書と対応づけるために持つ。
type Doc struct {
	ID      string
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
	totalLen [numFields]int
	avgLen   [numFields]float64
	vocab    map[string]int32
	words    []string
	// trie は語彙を文字単位で持ち、接頭辞の一致と打ち間違いの許容で語彙を全件なめずに済ませる。完全一致は vocab で引く。
	trie     *vocabTrie
	postings [][]posting
	norms    fieldNorms
	idfLog2  bool

	docExpansion     *Thesaurus
	expansionDropped int
	d2q              *Doc2Query
	d2qFiltered      int
	// reduce は足した語の出現回数から引く値を、語と文書の組ごとに項目別に持つ。重みが 1 の語だけなら nil のままにする。
	reduce         map[uint64]*[numFields]float64
	addedPositions int
	// gen は Doc2Query.Field のときの、生成したクエリの項目での語と文書の組ごとの出現回数。genLen はその項目の文書ごとの長さで、Field でなければ nil。
	gen    map[uint64]int32
	genLen []int32
	genAvg float64

	gallopRatio int
	bounds      *boundCache
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

// WithLog2IDF は BM25 の IDF の対数の底を自然対数から 2 に変える。点数が 1/ln2 倍になるだけで順位は変わらない。
func WithLog2IDF() Option { return func(ix *Index) { ix.idfLog2 = true } }

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

		gallopRatio: defaultGallopRatio,
		bounds:      newBoundCache(),
	}
	for _, o := range opts {
		o(ix)
	}
	var generated [][]string
	if ix.d2q != nil {
		generated = ix.generateQueries(docs, opts)
		if ix.d2q.Field {
			ix.genLen = make([]int32, len(docs))
		}
	}
	for id, d := range docs {
		var queries []string
		if generated != nil {
			queries = generated[id]
		}
		ix.add(int32(id), d, queries, &ix.totalLen)
	}
	ix.avgLen = averageLength(ix.totalLen, len(docs))
	if ix.d2qField() {
		ix.genAvg = generatedAverage(ix.genLen)
	}
	ix.norms = ix.docNorms(len(docs), nil)
	ix.trie = newVocabTrie(ix.words)
	return ix
}

// add は 1 文書の位置を 1 本の配列にまとめて確保し、語ごとの postings はその部分列を指す。
func (ix *Index) termID(term string) int32 {
	if id, ok := ix.vocab[term]; ok {
		return id
	}
	id := int32(len(ix.postings))
	ix.vocab[term] = id
	ix.words = append(ix.words, term)
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

func (ix *Index) add(id int32, d Doc, queries []string, total *[numFields]int) {
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
		if extra := ix.extraTokens(tokens[f], Field(f), queries); len(extra) > 0 {
			tokens[f] = mergeTokens(tokens[f], extra)
			ix.addedPositions += len(extra)
			for _, e := range extra {
				if e.generated && ix.d2qField() {
					ix.addGenerated(ix.termID(e.tok.term), id)
					continue
				}
				if e.weight != 1 {
					ix.reduceTF(ix.termID(e.tok.term), id, Field(f), 1-e.weight)
				}
			}
		}
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

// Stats の Positions は足した語の位置も含み、AddedPositions はそのうち文書拡張と doc2query で足した数。
// ExpansionDropped は文書拡張で 1 語あたりの上限を超えて捨てた語の数、Doc2QueryFiltered は doc2query の点数の下限で捨てたクエリの数。
type Stats struct {
	Docs              int
	Terms             int
	Postings          int
	Positions         int
	AddedPositions    int
	ExpansionDropped  int
	Doc2QueryFiltered int
}

func (ix *Index) Stats() Stats {
	s := Stats{
		Docs: len(ix.docs), Terms: len(ix.vocab),
		AddedPositions: ix.addedPositions, ExpansionDropped: ix.expansionDropped, Doc2QueryFiltered: ix.d2qFiltered,
	}
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
// Expanded は Query.Thesaurus で足した語の数、ExpansionDropped は 1 語あたりの上限を超えて捨てた語の数。どちらも元のクエリの書き方で数える。
type Result struct {
	Hits             []Hit
	Scored           int
	Expanded         int
	ExpansionDropped int
	MatchTime        time.Duration
	RankTime         time.Duration
}

// Query の Fields を空にするとすべての項目を検索する。Operator の既定は OperatorOr。
// Phrase はクエリの索引語が同じ項目で同じ間隔で並ぶ文書だけを残す。すべての語を含むことが前提なので Operator によらない。
// Synonyms は検索のときだけクエリを広げる辞書で、見出しと置き換え先のどちらかを含むクエリを両方の書き方の OR にする。
// Typo は英数字の語に打ち間違いを許し、Prefix はクエリの最後の語を接頭辞としても一致させる。Ranking の既定は RankingBM25。TFIDF は RankingTFIDF と RankingTFIDFCosine のときだけ使う。
// Near が正なら、クエリの語がすべて同じ項目で、順番を問わず最初と最後の位置の差が Near 以内に現れる文書だけを残す（NEAR/k）。Phrase を優先する。
// Expr が nil でなければ候補を論理式で決め、Text / Operator / Phrase / Near を使わない。採点は式の中の語すべてで行う。
// Thesaurus はクエリの語に類語辞書のグループの語と狭い語を足す。足した語ごとに点数を求め、元の語と足した語のうち最も高いものを取る。
// Extra は採点に足す索引語と重みで、適合フィードバックの展開語に使う。語の点数に重みを掛けて足す。
// Pruning は上位 limit 件に入りえない文書の採点を省く。BlockSize は PruningBlockMaxWAND の区間の件数で、0 なら DefaultBlockSize。
// Thesaurus と Extra の語は OperatorOr の候補を広げるが、OperatorAnd / Phrase / Near / Expr の候補は元の語だけで決め、採点にだけ使う。RankingBucket では使わない。
type Query struct {
	Text     string
	Expr     *Expr
	Fields   []Field
	Operator Operator
	Phrase   bool
	Near     int
	Synonyms map[string]string
	Typo     bool
	Prefix   bool
	Ranking  Ranking
	TFIDF    TFIDF

	Thesaurus *Thesaurus
	Extra     []WeightedTerm

	Pruning   Pruning
	BlockSize int
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
	res, _ := ix.rankWith(q, limit, nil)
	return res
}

// corpus は採点に使う文書全体の統計。df は語ごとの文書頻度で、nil なら索引の中で数える。idf は TF-IDF の IDF を語ごとに与えるときに使う。
// norms はコサインの文書ベクトルの長さで、nil なら索引を作ったときの値を使う。
type corpus struct {
	docs   int
	avgLen [numFields]float64
	df     map[string]int
	idf    map[string]float64
	norms  *fieldNorms
}

func averageLength(total [numFields]int, docs int) [numFields]float64 {
	var avg [numFields]float64
	if docs > 0 {
		for f := range numFields {
			avg[f] = float64(total[f]) / float64(docs)
		}
	}
	return avg
}

// rankWith は Hits と同じ並びで索引の中の文書番号も返す。
func (ix *Index) rankWith(q Query, limit int, c *corpus) (Result, []int) {
	start := time.Now()
	sel := selectFields(q.Fields)
	q = q.scoringText()
	pq := ix.parse(q)
	df := ix.docFreq(pq.terms, sel)
	if c == nil {
		c = &corpus{docs: len(ix.docs), avgLen: ix.avgLen}
	} else {
		for i, w := range pq.words {
			df[i] = c.df[w]
		}
	}
	if eligibleForPruning(q, pq, limit) {
		return ix.rankPruned(q, pq, df, sel, limit, c, start)
	}
	var (
		candidates []int
		masks      map[int]uint64
	)
	switch {
	case q.Expr != nil:
		candidates = ix.matchExpr(q.Expr, sel)
	case q.Phrase:
		candidates, masks = ix.matchAll(pq, sel, func(id int, v variant) bool { return ix.phraseIn(id, pq, v, sel) })
	case q.Near > 0:
		candidates, masks = ix.matchAll(pq, sel, func(id int, v variant) bool { return ix.nearIn(id, pq, v, sel, q.Near) })
	case q.Operator == OperatorAnd:
		candidates, masks = ix.matchAll(pq, sel, nil)
	default:
		candidates = ix.match(pq.terms, sel)
	}
	matched := time.Now()
	contrib := make([]float64, len(pq.terms))
	var idfs []float64
	tfidf := q.TFIDF
	if q.Ranking == RankingTFIDFCosine {
		tfidf = TFIDF{NormalizeQuery: q.TFIDF.NormalizeQuery}
	}
	if q.Ranking == RankingTFIDF || q.Ranking == RankingTFIDFCosine {
		idfs = tfidfWeights(tfidf, pq, df, c)
	}
	qnorm := 1.0
	var norms []float64
	if q.Ranking == RankingTFIDFCosine {
		if tfidf.NormalizeQuery {
			qnorm = queryNorm(idfs)
		}
		norms = ix.norms[sel.bits()]
		if c.norms != nil {
			norms = c.norms[sel.bits()]
		}
	}
	top := newRankTop(len(candidates), limit)
	for i, id := range candidates {
		mask := ^uint64(0)
		if masks != nil {
			mask = masks[id]
		}
		var s float64
		switch q.Ranking {
		case RankingBucket:
			s = ix.bucketScore(id, pq, sel, mask)
		case RankingTFIDF:
			s = ix.tfidfScore(id, pq, idfs, sel, mask, contrib, tfidf.TF, nil)
		case RankingTFIDFCosine:
			s = ix.tfidfScore(id, pq, idfs, sel, mask, contrib, tfidf.TF, norms) / qnorm
		default:
			s = ix.score(id, pq, df, sel, mask, contrib, c)
		}
		top.push(scored{ord: i, score: s})
	}
	ranked := top.result()
	hits := make([]Hit, len(ranked))
	ids := make([]int, len(ranked))
	for i, r := range ranked {
		id := candidates[r.ord]
		hits[i] = Hit{Doc: ix.docs[id], Score: r.score}
		ids[i] = id
	}
	return Result{
		Hits: hits, Scored: len(candidates), Expanded: pq.expanded, ExpansionDropped: pq.dropped,
		MatchTime: matched.Sub(start), RankTime: time.Since(matched),
	}, ids
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

func fieldSetOf(bits int) fieldSet {
	var s fieldSet
	for f := range numFields {
		s[f] = bits&(1<<f) != 0
	}
	return s
}

func (s fieldSet) bits() int {
	b := 0
	for f := range numFields {
		if s[f] {
			b |= 1 << f
		}
	}
	return b
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

// match はクエリ語の postings の和集合を、文書番号の列を前から併合して返す。どの語も含まない文書はここで落ち、採点されない。
// マップで重複を除いて並べ替えるより、BenchmarkMatchOr で 5 から 25 倍速かった。
func (ix *Index) match(terms []int32, sel fieldSet) []int {
	all := sel == selectFields(nil)
	var ids []int32
	for _, t := range terms {
		ids = unionIDs(ids, docsOf(ix.postingsOf(t), sel, all))
	}
	out := make([]int, len(ids))
	for i, id := range ids {
		out[i] = int(id)
	}
	return out
}

func (ix *Index) phraseIn(id int, pq parsedQuery, v variant, sel fieldSet) bool {
	var (
		psBuf    [4][]posting
		listsBuf [4][]int32
		offsBuf  [4]int32
	)
	ps := psBuf[:0]
	for _, slot := range v.slots {
		p := ix.present(id, pq, slot)
		if len(p) == 0 {
			return false
		}
		ps = append(ps, p)
	}
	lists, offs := listsBuf[:0], offsBuf[:0]
	for _, s := range v.seq {
		lists = append(lists, nil)
		offs = append(offs, s.off)
	}
	for f := range numFields {
		if !sel[f] {
			continue
		}
		for i, s := range v.seq {
			lists[i] = slotPositions(ps[s.slot], Field(f))
		}
		if ok, _ := phraseMatch(lists, offs); ok {
			return true
		}
	}
	return false
}

// slotPositions は slot の索引語のうち文書に現れるものの、項目 f での位置を昇順に並べる。
func slotPositions(ps []posting, f Field) []int32 {
	if len(ps) == 1 {
		return ps[0].positions(f)
	}
	var out []int32
	for _, p := range ps {
		out = append(out, p.positions(f)...)
	}
	slices.Sort(out)
	return out
}

// phraseMatch は先頭の語の位置を起点の候補にし、語ごとに起点+間隔と位置の列を 2 つのポインタで前から突き合わせて候補を絞る。最後の語で 1 つ一致すれば true。比べた回数も返す。
func phraseMatch(lists [][]int32, offs []int32) (bool, int) {
	bases := lists[0]
	var buf []int32
	n := 0
	for k := 1; k < len(lists); k++ {
		last := k == len(lists)-1
		if !last && buf == nil {
			buf = make([]int32, 0, len(bases))
		}
		next, off := lists[k], offs[k]
		out := buf[:0]
		i, j := 0, 0
		for i < len(bases) && j < len(next) {
			n++
			switch a, b := bases[i]+off, next[j]; {
			case a < b:
				i++
			case a > b:
				j++
			default:
				if last {
					return true, n
				}
				out = append(out, bases[i])
				i++
				j++
			}
		}
		if len(out) == 0 {
			return false, n
		}
		bases = out
	}
	return len(bases) > 0, n
}

// present は slot の索引語のうち文書に現れるものの postings を返す。
func (ix *Index) present(id int, pq parsedQuery, slot []alt) []posting {
	var ps []posting
	for _, a := range slot {
		if p, ok := ix.lookup(pq.terms[a.term], id); ok {
			ps = append(ps, p)
		}
	}
	return ps
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
func (ix *Index) score(id int, pq parsedQuery, df []int, sel fieldSet, mask uint64, contrib []float64, c *corpus) float64 {
	for i, t := range pq.terms {
		contrib[i] = 0
		if df[i] == 0 {
			continue
		}
		p, ok := ix.lookup(t, id)
		if !ok {
			continue
		}
		tf := ix.normalizedTF(t, id, p, sel, c.avgLen)
		if tf == 0 {
			continue
		}
		contrib[i] = ix.idf(c.docs, df[i]) * tf * (ix.k1 + 1) / (tf + ix.k1)
	}
	return combine(pq, mask, contrib)
}

// combine は語ごとの点数から、書き方ごとに語の位置ごとの最大を足し、最も高い書き方の値に Extra の語の点数を重みを掛けて足す。
func combine(pq parsedQuery, mask uint64, contrib []float64) float64 {
	best := 0.0
	for vi, v := range pq.variants {
		if mask&(1<<vi) == 0 {
			continue
		}
		if v.grouped != nil {
			best = max(best, v.groupedScore(contrib))
			continue
		}
		s := 0.0
		for _, slot := range v.slots {
			s += slotScore(slot, contrib)
		}
		best = max(best, s)
	}
	for _, e := range pq.extra {
		best += e.weight * contrib[e.term]
	}
	return best
}

// normalizedTF は出現回数を文書の長さで割る。tf/norm で置けば BM25 の tf*(k1+1)/(tf+k1*norm) と同じ値になる。
func (ix *Index) normalizedTF(term int32, id int, p posting, sel fieldSet, avgLen [numFields]float64) float64 {
	red := ix.reduction(term, p.doc)
	field := ix.d2qField()
	if ix.weights == nil && !field {
		var tf, length int
		var avg, r float64
		for f := range numFields {
			if sel[f] {
				tf += p.tf(f)
				length += ix.lengths[id][f]
				avg += avgLen[f]
				if red != nil {
					r += red[f]
				}
			}
		}
		return (float64(tf) - r) / (1 - ix.b + ix.b*float64(length)/avg)
	}
	weights := [numFields]float64{1, 1}
	if ix.weights != nil {
		weights = *ix.weights
	}
	var g float64
	if field && sel[FieldContent] {
		g = ix.generatedTF(term, p.doc)
	}
	s := 0.0
	for f := range numFields {
		tf := p.tf(f)
		if !sel[f] || tf == 0 {
			continue
		}
		x := float64(tf)
		if red != nil {
			x -= red[f]
		}
		if Field(f) == FieldContent {
			x -= g
		}
		norm := 1 - ix.b + ix.b*float64(ix.lengths[id][f])/avgLen[f]
		s += weights[f] * x / norm
	}
	if g > 0 {
		s += ix.d2q.Weight * g / (1 - ix.b + ix.b*float64(ix.genLen[id])/ix.genAvg)
	}
	return s
}

func reduceKey(term, doc int32) uint64 { return uint64(uint32(term))<<32 | uint64(uint32(doc)) }

func (ix *Index) reduceTF(term, doc int32, f Field, by float64) {
	if ix.reduce == nil {
		ix.reduce = make(map[uint64]*[numFields]float64)
	}
	k := reduceKey(term, doc)
	r, ok := ix.reduce[k]
	if !ok {
		r = new([numFields]float64)
		ix.reduce[k] = r
	}
	r[f] += by
}

// reduction は足した語の出現回数から引く値を返す。足した語が無ければ nil。
func (ix *Index) reduction(term, doc int32) *[numFields]float64 {
	if ix.reduce == nil {
		return nil
	}
	return ix.reduce[reduceKey(term, doc)]
}

// effectiveTF は選んだ項目の出現回数から、足した語の重みの分を引いた値を返す。
func (ix *Index) effectiveTF(term int32, p posting, sel fieldSet) float64 {
	x := float64(sel.tf(p))
	if ix.d2qField() && sel[FieldContent] {
		x -= ix.generatedTF(term, p.doc) * (1 - ix.d2q.Weight)
	}
	if red := ix.reduction(term, p.doc); red != nil {
		for f := range numFields {
			if sel[f] {
				x -= red[f]
			}
		}
	}
	return x
}

// extraTokens は索引を作るときに項目 f に足す語を返す。文書拡張の語は項目ごとに、生成したクエリの語は本文の後ろに足す。
func (ix *Index) extraTokens(tokens []token, f Field, queries []string) []weightedToken {
	var extra []weightedToken
	if ix.docExpansion != nil {
		extra = ix.expandTokens(tokens)
	}
	if f == FieldContent && len(queries) > 0 {
		last := int32(-d2qGap)
		for _, t := range tokens {
			last = max(last, t.pos)
		}
		for _, e := range extra {
			last = max(last, e.tok.pos)
		}
		extra = append(extra, ix.queryTokens(queries, last+d2qGap)...)
	}
	return extra
}

func (ix *Index) idf(docs, df int) float64 {
	n := float64(docs)
	v := math.Log(1 + (n-float64(df)+0.5)/(float64(df)+0.5))
	if ix.idfLog2 {
		return v / math.Ln2
	}
	return v
}
