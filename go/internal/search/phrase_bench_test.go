package search_test

import (
	"fmt"
	"testing"
	"time"

	"github.com/hiro8ma/agent/go/internal/search"
)

// BenchmarkPhrase はフレーズの候補を絞る時間を、語の頻度と語の数を変えて測る。
//
//	go test -run '^$' -bench Phrase ./internal/search/
func BenchmarkPhrase(b *testing.B) {
	corpora := []struct {
		name    string
		docs    func(int) []search.Doc
		queries map[string]string
	}{
		{name: "latin", docs: randomDocs, queries: map[string]string{
			"common2": "w00001 w00002",
			"common3": "w00001 w00002 w00003",
			"rare2":   benchQuery,
		}},
		{name: "japanese", docs: randomJapaneseDocs, queries: map[string]string{
			"word": kanjiWord(1) + "の" + kanjiWord(2),
		}},
	}
	for _, c := range corpora {
		for _, n := range []int{10_000, 100_000} {
			ix := search.New(c.docs(n))
			for qn, text := range c.queries {
				b.Run(fmt.Sprintf("%s/%s/N=%d", c.name, qn, n), func(b *testing.B) {
					q := search.Query{Text: text, Phrase: true}
					var scored int
					var match time.Duration
					for b.Loop() {
						res := ix.RankQuery(q, 10)
						scored = res.Scored
						match += res.MatchTime
					}
					b.ReportMetric(float64(scored), "scored/op")
					b.ReportMetric(float64(match.Nanoseconds())/float64(b.N), "match-ns/op")
				})
			}
		}
	}
}
