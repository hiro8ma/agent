package search

import "slices"

// QueryGenerator は文書から、その文書を探す人が打ちそうなクエリを作る。doc2query のモデルの代わりに差し替える口。
type QueryGenerator func(d Doc) []string

// Doc2Query は索引を作るときに、生成したクエリの語を本文の後ろに足す。足した語は文書の長さに数えず、出現 1 回を Weight 倍で数える。Weight が 0 なら 1。
// MinScore が正なら、生成したクエリを元の文書（生成したクエリを足す前の索引）に BM25 で採点し、MinScore 未満のものを捨てる（doc2query--）。
// 元の文書に無い語だけから成るクエリは 0 点になり、生成の誤りで無関係な語が文書に付くのを防げる。
// Field なら生成したクエリを本文と別の項目として数え、BM25 は項目ごとに長さを正規化して Weight を項目の重みに掛ける（BM25F）。
// タイトルと本文の重みは WithFieldWeights の値で、指定しなければ 1。生成した語の出現は本文の出現回数から除き、生成したクエリの語の数を項目の長さにする。
// Fields で本文を選ぶと生成したクエリの項目も含む。BM25 以外の採点では Field でない場合と同じく出現 1 回を Weight 倍で数える。
type Doc2Query struct {
	Generate QueryGenerator
	Weight   float64
	MinScore float64
	Field    bool
}

func WithDoc2Query(d Doc2Query) Option {
	return func(ix *Index) {
		if d.Generate == nil {
			ix.d2q = nil
			return
		}
		if d.Weight == 0 {
			d.Weight = 1
		}
		ix.d2q = &d
	}
}

// d2qGap は生成したクエリどうしの位置の間隔。クエリをまたいでフレーズが一致しないように 1 つ空ける。
const d2qGap = 2

// generateQueries は文書ごとに残した生成クエリを返す。MinScore が正なら、生成したクエリを足さずに作った索引で採点する。
func (ix *Index) generateQueries(docs []Doc, opts []Option) [][]string {
	out := make([][]string, len(docs))
	var base *Index
	if ix.d2q.MinScore > 0 {
		base = New(docs, append(slices.Clone(opts), WithDoc2Query(Doc2Query{}))...)
	}
	for id, d := range docs {
		for _, q := range ix.d2q.Generate(d) {
			if base != nil && base.scoreDoc(Query{Text: q}, id) < ix.d2q.MinScore {
				ix.d2qFiltered++
				continue
			}
			out[id] = append(out[id], q)
		}
	}
	return out
}

// scoreDoc は文書 1 件に対するクエリの BM25 の点数を返す。
func (ix *Index) scoreDoc(q Query, id int) float64 {
	sel := selectFields(q.Fields)
	pq := ix.parse(q)
	df := ix.docFreq(pq.terms, sel)
	contrib := make([]float64, len(pq.terms))
	return ix.score(id, pq, df, sel, ^uint64(0), contrib, &corpus{docs: len(ix.docs), avgLen: ix.avgLen})
}

// queryTokens は生成したクエリを索引語にし、位置 start から並べる。
func (ix *Index) queryTokens(queries []string, start int32) []weightedToken {
	next := start
	var out []weightedToken
	for _, q := range queries {
		tokens := ix.analyzer.analyze(q)
		for _, t := range tokens {
			out = append(out, weightedToken{tok: token{term: t.term, pos: next + t.pos}, weight: ix.d2q.Weight, generated: true})
		}
		if len(tokens) > 0 {
			next += tokens[len(tokens)-1].pos + d2qGap
		}
	}
	return out
}

// d2qField は生成したクエリを別の項目として採点するか。
func (ix *Index) d2qField() bool { return ix.genLen != nil }

func (ix *Index) addGenerated(term, doc int32) {
	if ix.gen == nil {
		ix.gen = make(map[uint64]int32)
	}
	ix.gen[reduceKey(term, doc)]++
	ix.genLen[doc]++
}

// generatedTF は文書 doc の生成したクエリの項目での語 term の出現回数を返す。
func (ix *Index) generatedTF(term, doc int32) float64 {
	if ix.gen == nil {
		return 0
	}
	return float64(ix.gen[reduceKey(term, doc)])
}

// generatedAverage は生成したクエリの項目の平均の長さを、その項目を持つ文書だけで割って求める。Lucene の BM25 が項目を持つ文書の数で割るのと同じ。
// 全文書で割ると、生成したクエリを持つ文書が少ないほど平均が 0 に近づき、持つ文書の長さの正規化が数十倍に効いて点数がほぼ 0 になる。
func generatedAverage(lengths []int32) float64 {
	var total, docs int
	for _, n := range lengths {
		if n > 0 {
			total += int(n)
			docs++
		}
	}
	if docs == 0 {
		return 0
	}
	return float64(total) / float64(docs)
}
