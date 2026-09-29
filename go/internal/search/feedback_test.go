package search_test

import (
	"slices"
	"strings"
	"testing"

	"github.com/hiro8ma/agent/go/internal/search"
)

func TestFeedbackTermsWeights(t *testing.T) {
	t.Parallel()
	ds := evalDocs()
	ix := search.New(ds, search.WithAnalyzer(evalAnalyzer()))
	var fruit []int
	for i, d := range ds {
		if strings.HasPrefix(d.ID, "fruit") {
			fruit = append(fruit, i)
		}
	}
	terms := ix.FeedbackTerms(search.Query{Text: "apple"}, fruit, search.Feedback{Terms: 3, Weight: 0.4})
	if len(terms) != 3 || terms[0].Term != "fruit" || terms[0].Weight != 0.4 {
		t.Fatalf("terms = %+v, want fruit first with weight 0.4", terms)
	}
	for i, w := range terms {
		if w.Term == "appl" || (i > 0 && w.Weight > terms[i-1].Weight) {
			t.Fatalf("terms = %+v, want query term excluded and weights descending", terms)
		}
	}
}

// TestPseudoRelevanceFeedbackDrift は 1 ページ目が的外れなクエリで、擬似適合フィードバックが的外れな話題の語を足して nDCG を下げることと、その緩和を確かめる。
func TestPseudoRelevanceFeedbackDrift(t *testing.T) {
	t.Parallel()
	s := newEvalSetup()
	qs := evalQueries()
	ndcg := func(q evalQuery, fb *search.Feedback) float64 {
		if fb == nil {
			return measure(s.plain.RankQuery(search.Query{Text: q.text}, evalK).Hits, q.relevant).ndcg
		}
		r, _ := s.plain.PseudoRelevanceFeedback(search.Query{Text: q.text}, *fb, evalK)
		return measure(r.Hits, q.relevant).ndcg
	}
	testCases := map[string]struct {
		query      string
		drift      search.Feedback
		mitigation search.Feedback
	}{
		"apple の上位 3 件が会社の文書で、2 件以上に現れる語だけを足すと下がらない": {
			query:      "apple",
			drift:      search.Feedback{Docs: 3, Terms: 20},
			mitigation: search.Feedback{Docs: 3, Terms: 20, MinDocs: 2},
		},
		"apple の上位 3 件が会社の文書で、足す語の重みを下げると下がり方が小さい": {
			query:      "apple",
			drift:      search.Feedback{Docs: 3, Terms: 20},
			mitigation: search.Feedback{Docs: 3, Terms: 20, Weight: 0.25},
		},
		"iphone battery の上位 10 件に会社の文書が混ざり、1 位の半分以上の点数の文書だけを使うと下がらない": {
			query:      "iphone battery",
			drift:      search.Feedback{Docs: 10, Terms: 10},
			mitigation: search.Feedback{Docs: 10, Terms: 10, MinTopShare: 0.5},
		},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			q := qs[tc.query]
			base, drifted, mitigated := ndcg(q, nil), ndcg(q, &tc.drift), ndcg(q, &tc.mitigation)
			if drifted >= base {
				t.Fatalf("nDCG base %.3f, PRF %.3f, want PRF lower", base, drifted)
			}
			if mitigated <= drifted {
				t.Fatalf("nDCG PRF %.3f, mitigated %.3f, want mitigated higher", drifted, mitigated)
			}
			t.Logf("nDCG@10 base %.3f, PRF %.3f, mitigated %.3f", base, drifted, mitigated)
		})
	}
}

func TestRelevanceFeedbackResolvesAmbiguity(t *testing.T) {
	t.Parallel()
	s := newEvalSetup()
	q := evalQueries()["apple"]
	base := measure(s.plain.RankQuery(search.Query{Text: q.text}, evalK).Hits, q.relevant)
	r, terms := s.plain.RelevanceFeedback(search.Query{Text: q.text}, []string{"fruit1", "fruit3"}, search.Feedback{}, evalK)
	got := measure(r.Hits, q.relevant)
	if got.ndcg <= base.ndcg || got.recall < base.recall {
		t.Fatalf("nDCG %.3f -> %.3f, recall %.2f -> %.2f", base.ndcg, got.ndcg, base.recall, got.recall)
	}
	if !slices.ContainsFunc(terms, func(w search.WeightedTerm) bool { return w.Term == "fruit" }) {
		t.Fatalf("terms = %+v, want fruit", terms)
	}
}

func TestExtraWeightScalesTermScore(t *testing.T) {
	t.Parallel()
	ix := search.New(evalDocs(), search.WithAnalyzer(evalAnalyzer()))
	score := func(w float64) float64 {
		hits := ix.RankQuery(search.Query{Text: "fruit", Extra: []search.WeightedTerm{{Term: "orchard", Weight: w}}}, -1).Hits
		for _, h := range hits {
			if h.Doc.ID == "fruit5" {
				return h.Score
			}
		}
		return 0
	}
	base := ix.RankQuery(search.Query{Text: "fruit"}, -1).Hits
	var fruit float64
	for _, h := range base {
		if h.Doc.ID == "fruit5" {
			fruit = h.Score
		}
	}
	orchard := score(1) - fruit
	if got := score(0.25) - fruit; orchard <= 0 || !near(got, 0.25*orchard) {
		t.Fatalf("orchard part at weight 0.25 = %v, want %v", got, 0.25*orchard)
	}
}

func near(a, b float64) bool {
	d := a - b
	return d < 1e-9 && d > -1e-9
}
