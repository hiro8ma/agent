package search

import "math"

// TFWeight は TF-IDF で出現回数 f を重みに変える方法。既定の TFLog は 1 + log10(f)。
type TFWeight int

const (
	TFLog TFWeight = iota
	TFRaw
	TFSqrt
)

func (w TFWeight) weight(f float64) float64 {
	if f <= 0 {
		return 0
	}
	switch w {
	case TFRaw:
		return f
	case TFSqrt:
		return math.Sqrt(f)
	default:
		// 重みを下げて足した語の 1 未満の回数で負にならないように、1 未満は回数のまま使う。1 で 1 + log10(1) とつながる。
		if f < 1 {
			return f
		}
		return 1 + math.Log10(f)
	}
}

// IDFWeight は文書数 N と文書頻度 DF から IDF を求める方法。既定の IDFSmooth は log2(1 + N/DF)、IDFPlain は log2(N/DF)。
type IDFWeight int

const (
	IDFSmooth IDFWeight = iota
	IDFPlain
)

func (w IDFWeight) weight(docs, df int) float64 {
	if df <= 0 || docs <= 0 {
		return 0
	}
	r := float64(docs) / float64(df)
	if w == IDFPlain {
		return math.Log2(r)
	}
	return math.Log2(1 + r)
}

// TFIDF は RankingTFIDF の変種。点数はクエリの語ごとの TF × IDF の和。TF は項目の重み（WithFieldWeights）も文書の長さも使わない。
// NormalizeQuery は RankingTFIDFCosine でクエリのベクトルの長さでも割る。全文書で同じ値なので順位は変わらない。
type TFIDF struct {
	TF             TFWeight
	IDF            IDFWeight
	NormalizeQuery bool
}

// fieldNorms は文書ベクトルの長さを項目の組ごとに持つ。添字は fieldSet.bits で、0（項目なし）は使わない。
type fieldNorms [1 << numFields][]float64

// fieldDF は語の postings から、項目の組ごとにその組のどれかに語を含む文書の数を数える。
func fieldDF(ps []posting) [1 << numFields]int {
	var n [1 << numFields]int
	for _, p := range ps {
		m := presence(p)
		for set := range n {
			if set&m != 0 {
				n[set]++
			}
		}
	}
	return n
}

// presence は語を含む項目をビットで返す。
func presence(p posting) int {
	m := 0
	for f := range numFields {
		if p.tf(f) > 0 {
			m |= 1 << f
		}
	}
	return m
}

// docNorms は既定の TF と IDF の重みで、項目の組ごとに文書ベクトルの長さを求める。TF と DF は組の中の項目だけで数え、クエリの Fields で採点するときの分子とそろえる。
// docs と df は IDF に使う文書数と語ごとの文書頻度で、df が nil なら索引の中で数える。
func (ix *Index) docNorms(docs int, df map[string][1 << numFields]int) fieldNorms {
	var norms fieldNorms
	for set := 1; set < len(norms); set++ {
		norms[set] = make([]float64, len(ix.docs))
	}
	var (
		def  TFIDF
		idf  [1 << numFields]float64
		sets [1 << numFields]fieldSet
	)
	for set := range sets {
		sets[set] = fieldSetOf(set)
	}
	for t, ps := range ix.postings {
		var n [1 << numFields]int
		if df == nil {
			n = fieldDF(ps)
		} else {
			n = df[ix.words[t]]
		}
		for set := 1; set < len(idf); set++ {
			idf[set] = def.IDF.weight(docs, n[set])
		}
		for _, p := range ps {
			for set := 1; set < len(norms); set++ {
				if sets[set].tf(p) == 0 {
					continue
				}
				w := def.TF.weight(ix.effectiveTF(int32(t), p, sets[set])) * idf[set]
				norms[set][p.doc] += w * w
			}
		}
	}
	for set := 1; set < len(norms); set++ {
		for i, s := range norms[set] {
			norms[set][i] = math.Sqrt(s)
		}
	}
	return norms
}

// tfidfWeights はクエリの語ごとの IDF を返す。c.idf にある語はその値を使う。
func tfidfWeights(cfg TFIDF, pq parsedQuery, df []int, c *corpus) []float64 {
	idfs := make([]float64, len(pq.terms))
	for i, w := range pq.words {
		if v, ok := c.idf[w]; ok {
			idfs[i] = v
			continue
		}
		idfs[i] = cfg.IDF.weight(c.docs, df[i])
	}
	return idfs
}

// queryNorm はクエリの語の出現を 1 回として、IDF を並べたベクトルの長さを返す。
func queryNorm(idfs []float64) float64 {
	s := 0.0
	for _, v := range idfs {
		s += v * v
	}
	return math.Sqrt(s)
}

// tfidfScore は文書に現れるクエリの語の出現記録だけから TF × IDF を求め、BM25 と同じく書き方ごとの最大を返す。
// norms が nil でなければコサインにし、内積を文書のベクトルの長さで割る。クエリの重みは TF が 1 なので IDF と同じになる。
func (ix *Index) tfidfScore(id int, pq parsedQuery, idfs []float64, sel fieldSet, mask uint64, contrib []float64, tf TFWeight, norms []float64) float64 {
	cosine := norms != nil
	for i, t := range pq.terms {
		contrib[i] = 0
		if idfs[i] == 0 {
			continue
		}
		p, ok := ix.lookup(t, id)
		if !ok {
			continue
		}
		w := tf.weight(ix.effectiveTF(t, p, sel)) * idfs[i]
		if cosine {
			w *= idfs[i]
		}
		contrib[i] = w
	}
	s := combine(pq, mask, contrib)
	if cosine && s > 0 {
		s /= norms[id]
	}
	return s
}
