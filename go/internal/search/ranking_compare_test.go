package search_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/hiro8ma/agent/go/internal/search"
)

// randomDocsSpread は randomDocs と同じ語の分布で、長さを 5 から 500 語の対数一様にして長さのばらつきを大きくする。
func randomDocsSpread(n int) []search.Doc {
	r := rand.New(rand.NewPCG(43, uint64(n)))
	zipf := rand.NewZipf(r, 1.1, 1, 19_999)
	ds := make([]search.Doc, n)
	for i := range ds {
		words := make([]string, int(math.Round(5*math.Pow(100, r.Float64()))))
		for j := range words {
			words[j] = fmt.Sprintf("w%05d", zipf.Uint64())
		}
		ds[i] = search.Doc{Title: fmt.Sprintf("doc%d", i), Content: strings.Join(words, " ")}
	}
	return ds
}

// compareQueries は頻度の高い語と中くらいの語を組にした 20 個のクエリ。
func compareQueries() []search.Query {
	qs := make([]search.Query, 20)
	for i := range qs {
		qs[i] = search.Query{Text: fmt.Sprintf("w%05d w%05d", 5+i*7, 300+i*97)}
	}
	return qs
}

type rankingSummary struct {
	agree   float64
	meanLen float64
}

func topDocs(t testing.TB, ix *search.Index, q search.Query) []int {
	t.Helper()
	hits := ix.RankQuery(q, 10).Hits
	out := make([]int, len(hits))
	for i, h := range hits {
		if _, err := fmt.Sscanf(h.Doc.Title, "doc%d", &out[i]); err != nil {
			t.Fatal(err)
		}
	}
	return out
}

// summarize は base と比べた上位 10 件の一致率と、上位 10 件の平均文書長をクエリで平均する。
func summarize(t testing.TB, ix, base *search.Index, ranking search.Ranking, tfidf search.TFIDF) rankingSummary {
	t.Helper()
	var s rankingSummary
	var docs int
	qs := compareQueries()
	for _, q := range qs {
		want := make(map[int]bool)
		for _, d := range topDocs(t, base, q) {
			want[d] = true
		}
		q.Ranking, q.TFIDF = ranking, tfidf
		got := topDocs(t, ix, q)
		hit := 0
		for _, d := range got {
			if want[d] {
				hit++
			}
			s.meanLen += float64(search.DocLength(ix, d))
		}
		docs += len(got)
		s.agree += float64(hit) / float64(max(len(want), 1))
	}
	s.agree /= float64(len(qs))
	s.meanLen /= float64(max(docs, 1))
	return s
}

var (
	sweepK1 = []float64{0.5, 1.2, 2.0}
	sweepB  = []float64{0, 0.5, 0.75, 1}
)

var compareCorpora = []struct {
	name string
	docs func(int) []search.Doc
}{
	{name: "zipf", docs: randomDocs},
	{name: "spread", docs: randomDocsSpread},
}

func TestRankingLengthBias(t *testing.T) {
	t.Parallel()
	for _, c := range compareCorpora {
		t.Run(c.name, func(t *testing.T) {
			t.Parallel()
			ix := search.New(c.docs(2_000))
			bm25 := func(k1, b float64) rankingSummary {
				return summarize(t, search.WithBM25Of(ix, k1, b), ix, search.RankingBM25, search.TFIDF{})
			}
			tfidf := summarize(t, ix, ix, search.RankingTFIDF, search.TFIDF{})
			raw := summarize(t, ix, ix, search.RankingTFIDF, search.TFIDF{TF: search.TFRaw})
			cosine := summarize(t, ix, ix, search.RankingTFIDFCosine, search.TFIDF{})
			def := bm25(search.DefaultK1, search.DefaultB)
			testCases := map[string]struct {
				longer, shorter float64
			}{
				"b=0 は b=1 より長い文書を上位に置く":                {longer: bm25(1.2, 0).meanLen, shorter: bm25(1.2, 1).meanLen},
				"TF-IDF は BM25 より長い文書を上位に置く":            {longer: tfidf.meanLen, shorter: def.meanLen},
				"生の回数の TF-IDF は 1 + log10(f) より長い文書に偏る": {longer: raw.meanLen, shorter: tfidf.meanLen},
				"コサインで長さを割ると TF-IDF より短い文書が上位に来る":       {longer: tfidf.meanLen, shorter: cosine.meanLen},
			}
			for tn, tc := range testCases {
				t.Run(tn, func(t *testing.T) {
					t.Parallel()
					if tc.longer <= tc.shorter {
						t.Fatalf("mean length %v should exceed %v", tc.longer, tc.shorter)
					}
				})
			}
		})
	}
}

// BenchmarkBM25Params は k1 と b を変えたときの、既定（1.2, 0.75）との上位 10 件の一致率と上位 10 件の平均文書長を報告する。
func BenchmarkBM25Params(b *testing.B) {
	for _, c := range compareCorpora {
		ix := search.New(c.docs(10_000))
		for _, k1 := range sweepK1 {
			for _, bv := range sweepB {
				v := search.WithBM25Of(ix, k1, bv)
				b.Run(fmt.Sprintf("%s/N=10000/k1=%.1f/b=%.2f", c.name, k1, bv), func(b *testing.B) {
					s := summarize(b, v, ix, search.RankingBM25, search.TFIDF{})
					for b.Loop() {
						v.Rank(benchQuery, 10)
					}
					b.ReportMetric(s.agree, "agree@10")
					b.ReportMetric(s.meanLen, "len@10")
				})
			}
		}
	}
}

var rankingVariants = []struct {
	name    string
	ranking search.Ranking
	tfidf   search.TFIDF
}{
	{name: "bm25", ranking: search.RankingBM25},
	{name: "tfidf/log-smooth", ranking: search.RankingTFIDF},
	{name: "tfidf/log-plain", ranking: search.RankingTFIDF, tfidf: search.TFIDF{IDF: search.IDFPlain}},
	{name: "tfidf/raw-smooth", ranking: search.RankingTFIDF, tfidf: search.TFIDF{TF: search.TFRaw}},
	{name: "tfidf/raw-plain", ranking: search.RankingTFIDF, tfidf: search.TFIDF{TF: search.TFRaw, IDF: search.IDFPlain}},
	{name: "tfidf/sqrt-smooth", ranking: search.RankingTFIDF, tfidf: search.TFIDF{TF: search.TFSqrt}},
	{name: "tfidf/sqrt-plain", ranking: search.RankingTFIDF, tfidf: search.TFIDF{TF: search.TFSqrt, IDF: search.IDFPlain}},
	{name: "cosine", ranking: search.RankingTFIDFCosine},
}

// BenchmarkRankingVariants は TF-IDF の各変種とコサインの、BM25 との上位 10 件の一致率と上位 10 件の平均文書長と時間を報告する。
func BenchmarkRankingVariants(b *testing.B) {
	for _, c := range compareCorpora {
		for _, n := range []int{10_000, 100_000} {
			ix := search.New(c.docs(n))
			for _, v := range rankingVariants {
				b.Run(fmt.Sprintf("%s/N=%d/%s", c.name, n, v.name), func(b *testing.B) {
					s := summarize(b, ix, ix, v.ranking, v.tfidf)
					q := search.Query{Text: benchQuery, Ranking: v.ranking, TFIDF: v.tfidf}
					for b.Loop() {
						ix.RankQuery(q, 10)
					}
					b.ReportMetric(s.agree, "agree@10")
					b.ReportMetric(s.meanLen, "len@10")
				})
			}
		}
	}
}
