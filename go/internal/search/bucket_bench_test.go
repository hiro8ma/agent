package search_test

import (
	"fmt"
	"testing"

	"github.com/hiro8ma/agent/go/internal/search"
)

// BenchmarkRanking は BM25 と bucket sort で同じ候補を並べ、上位 10 件の一致率と 1 クエリの時間を出す。クエリは benchQuery と 2 語のクエリ 19 個。
//
//	go test -run '^$' -bench Ranking ./internal/search/
func BenchmarkRanking(b *testing.B) {
	queries := shardQueries(20, "")
	for _, n := range []int{10_000, 100_000} {
		ix := search.New(randomDocs(n))
		overlap := 0.0
		for _, q := range queries {
			bm25 := ix.RankQuery(search.Query{Text: q}, 10).Hits
			bucket := ix.RankQuery(search.Query{Text: q, Ranking: search.RankingBucket}, 10).Hits
			overlap += titleOverlap(bm25, bucket)
		}
		overlap /= float64(len(queries))
		for _, r := range []struct {
			name    string
			ranking search.Ranking
		}{{name: "bm25", ranking: search.RankingBM25}, {name: "bucket", ranking: search.RankingBucket}} {
			b.Run(fmt.Sprintf("N=%d/%s", n, r.name), func(b *testing.B) {
				i := 0
				scored := 0
				for b.Loop() {
					scored += ix.RankQuery(search.Query{Text: queries[i%len(queries)], Ranking: r.ranking}, 10).Scored
					i++
				}
				b.ReportMetric(float64(scored)/float64(b.N), "scored/op")
				b.ReportMetric(overlap, "overlap@10")
			})
		}
	}
}

func titleOverlap(want, got []search.Hit) float64 {
	if len(want) == 0 {
		return 1
	}
	seen := make(map[string]struct{}, len(got))
	for _, h := range got {
		seen[h.Doc.Title] = struct{}{}
	}
	n := 0
	for _, h := range want {
		if _, ok := seen[h.Doc.Title]; ok {
			n++
		}
	}
	return float64(n) / float64(len(want))
}
