package search_test

import (
	"fmt"
	"testing"
	"time"

	"github.com/hiro8ma/agent/go/internal/search"
)

// lostAt は OR の BM25 の上位 k 件のうち、AND の候補に入らない文書の数を返す。
func lostAt(ix *search.Index, text string, k int) int {
	and := map[string]struct{}{}
	for _, h := range ix.RankQuery(search.Query{Text: text, Operator: search.OperatorAnd}, -1).Hits {
		and[h.Doc.Title] = struct{}{}
	}
	lost := 0
	for _, h := range ix.RankQuery(search.Query{Text: text}, k).Hits {
		if _, ok := and[h.Doc.Title]; !ok {
			lost++
		}
	}
	return lost
}

// BenchmarkOperator は OR と AND の候補数と検索時間、AND で OR の上位 10 件から落ちる数を出す。AND は 2 つのポインタだけと、長さの比で galloping に切り替える場合を比べる。
//
//	go test -run '^$' -bench Operator ./internal/search/
func BenchmarkOperator(b *testing.B) {
	corpora := []struct {
		name    string
		docs    func(int) []search.Doc
		queries map[string]string
	}{
		{name: "latin", docs: randomDocs, queries: map[string]string{
			"rare":   benchQuery,
			"common": "w00001 w00002 w00003",
		}},
		{name: "japanese", docs: randomJapaneseDocs, queries: map[string]string{
			"particle": kanjiWord(50) + "には" + kanjiWord(2000),
			"spaced":   kanjiWord(50) + " " + kanjiWord(2000),
		}},
	}
	for _, c := range corpora {
		for _, n := range benchSizes {
			ix := search.New(c.docs(n))
			modes := []struct {
				name string
				ix   *search.Index
				op   search.Operator
			}{
				{name: "or", ix: ix, op: search.OperatorOr},
				{name: "and-merge", ix: search.WithGallopRatioOf(ix, 0), op: search.OperatorAnd},
				{name: "and-switch", ix: ix, op: search.OperatorAnd},
			}
			for qn, text := range c.queries {
				lost := lostAt(ix, text, 10)
				for _, m := range modes {
					b.Run(fmt.Sprintf("%s/%s/N=%d/%s", c.name, qn, n, m.name), func(b *testing.B) {
						q := search.Query{Text: text, Operator: m.op}
						var scored int
						var match time.Duration
						for b.Loop() {
							res := m.ix.RankQuery(q, 10)
							scored = res.Scored
							match += res.MatchTime
						}
						b.ReportMetric(float64(scored), "scored/op")
						b.ReportMetric(float64(match.Nanoseconds())/float64(b.N), "match-ns/op")
						if m.op == search.OperatorAnd {
							b.ReportMetric(float64(lost), "lost@10")
						}
					})
				}
			}
		}
	}
}
