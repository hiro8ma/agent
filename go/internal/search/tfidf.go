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

// docNorms は既定の TF と IDF の重みで文書のベクトルの長さを求める。IDF は索引を作った時点の N と DF で固定し、シャードでは各シャードの N と DF で決まる。
func (ix *Index) docNorms() []float64 {
	norms := make([]float64, len(ix.docs))
	var def TFIDF
	for _, ps := range ix.postings {
		idf := def.IDF.weight(len(ix.docs), len(ps))
		for _, p := range ps {
			w := def.TF.weight(float64(len(p.pos))) * idf
			norms[p.doc] += w * w
		}
	}
	for i, s := range norms {
		norms[i] = math.Sqrt(s)
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
// cosine なら内積を文書のベクトルの長さで割る。クエリの重みは TF が 1 なので IDF と同じになる。
func (ix *Index) tfidfScore(id int, pq parsedQuery, idfs []float64, sel fieldSet, mask uint64, contrib []float64, tf TFWeight, cosine bool) float64 {
	for i, t := range pq.terms {
		contrib[i] = 0
		if idfs[i] == 0 {
			continue
		}
		p, ok := ix.lookup(t, id)
		if !ok {
			continue
		}
		w := tf.weight(float64(sel.tf(p))) * idfs[i]
		if cosine {
			w *= idfs[i]
		}
		contrib[i] = w
	}
	s := combine(pq, mask, contrib)
	if cosine && s > 0 {
		s /= ix.norms[id]
	}
	return s
}
