package search

import (
	"cmp"
	"slices"
)

const (
	DefaultFeedbackDocs   = 10
	DefaultFeedbackTerms  = 10
	DefaultFeedbackWeight = 0.5
)

// Feedback は適合フィードバックでクエリに足す語の選び方。0 の項目は既定値を使う。
// Docs は擬似適合フィードバックで適合とみなす上位の件数（K）、Terms は足す語の数（M）。
// Weight は最も重い展開語の重みで、残りの展開語は語の重みの比で小さくする。元のクエリの語の重みは 1 のまま。
// MinDocs は展開語を含むフィードバック文書の数の下限。1 件の文書にしか無い語は、その文書の話題にクエリを引っ張りやすい。
// MinTopShare が正なら、擬似適合フィードバックで 1 回目の検索の上位 Docs 件のうち、1 位の点数の MinTopShare 倍以上の文書だけを適合とみなす。
type Feedback struct {
	Docs        int
	Terms       int
	Weight      float64
	MinDocs     int
	MinTopShare float64
}

func (fb Feedback) withDefaults() Feedback {
	if fb.Docs <= 0 {
		fb.Docs = DefaultFeedbackDocs
	}
	if fb.Terms <= 0 {
		fb.Terms = DefaultFeedbackTerms
	}
	if fb.Weight == 0 {
		fb.Weight = DefaultFeedbackWeight
	}
	fb.MinDocs = max(fb.MinDocs, 1)
	return fb
}

// FeedbackTerms は索引の中の文書番号 docs からクエリに足す語を選ぶ。
// 語の重みは Rocchio の適合文書の重心と同じく、フィードバック文書ごとの TF-IDF（TFLog × IDFSmooth）を平均したもの。
// 文書の語は元の本文とタイトルから数え、文書拡張や doc2query で足した語は使わない。クエリ（類語辞書と Extra を含む）の語は除く。
func (ix *Index) FeedbackTerms(q Query, docs []int, fb Feedback) []WeightedTerm {
	fb = fb.withDefaults()
	if len(docs) == 0 {
		return nil
	}
	exclude := make(map[string]bool)
	for _, w := range ix.parse(q.scoringText()).words {
		exclude[w] = true
	}
	type stat struct {
		sum  float64
		docs int
	}
	stats := make(map[string]*stat)
	for _, id := range docs {
		tf := make(map[string]int)
		for f := range numFields {
			for _, t := range ix.analyzer.analyze(ix.docs[id].field(f)) {
				tf[t.term]++
			}
		}
		for w, n := range tf {
			if exclude[w] {
				continue
			}
			s, ok := stats[w]
			if !ok {
				s = &stat{}
				stats[w] = s
			}
			s.sum += TFLog.weight(float64(n))
			s.docs++
		}
	}
	var def TFIDF
	out := make([]WeightedTerm, 0, len(stats))
	for w, s := range stats {
		if s.docs < fb.MinDocs {
			continue
		}
		idf := def.IDF.weight(len(ix.docs), len(ix.postingsOf(ix.lookupTerm(w))))
		out = append(out, WeightedTerm{Term: w, Weight: s.sum / float64(len(docs)) * idf})
	}
	slices.SortFunc(out, func(a, b WeightedTerm) int {
		return cmp.Or(cmp.Compare(b.Weight, a.Weight), cmp.Compare(a.Term, b.Term))
	})
	if len(out) > fb.Terms {
		out = out[:fb.Terms]
	}
	if len(out) > 0 && out[0].Weight > 0 {
		top := out[0].Weight
		for i := range out {
			out[i].Weight = fb.Weight * out[i].Weight / top
		}
	}
	return out
}

// RelevanceFeedback は利用者が適合と選んだ文書（Doc.ID）から語を選んでクエリに足し、もう一度検索する。足した語も返す。
func (ix *Index) RelevanceFeedback(q Query, relevant []string, fb Feedback, limit int) (Result, []WeightedTerm) {
	want := make(map[string]bool, len(relevant))
	for _, id := range relevant {
		want[id] = true
	}
	var docs []int
	for i, d := range ix.docs {
		if want[d.ID] {
			docs = append(docs, i)
		}
	}
	return ix.searchExpanded(q, docs, fb, limit)
}

// PseudoRelevanceFeedback は 1 回目の検索の上位 fb.Docs 件を適合とみなして語を選び、クエリに足してもう一度検索する（擬似適合フィードバック）。
// 上位が的外れなら、的外れな話題の語を足してさらに外れる（クエリドリフト）。
func (ix *Index) PseudoRelevanceFeedback(q Query, fb Feedback, limit int) (Result, []WeightedTerm) {
	fb = fb.withDefaults()
	first, ids := ix.rankWith(q, fb.Docs, nil)
	if fb.MinTopShare > 0 && len(first.Hits) > 0 {
		floor := first.Hits[0].Score * fb.MinTopShare
		n := 0
		for n < len(first.Hits) && first.Hits[n].Score >= floor {
			n++
		}
		ids = ids[:n]
	}
	return ix.searchExpanded(q, ids, fb, limit)
}

func (ix *Index) searchExpanded(q Query, docs []int, fb Feedback, limit int) (Result, []WeightedTerm) {
	terms := ix.FeedbackTerms(q, docs, fb)
	q.Extra = append(slices.Clone(q.Extra), terms...)
	return ix.RankQuery(q, limit), terms
}
