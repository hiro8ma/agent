package search_test

import (
	"fmt"
	"testing"

	"github.com/hiro8ma/agent/go/internal/search"
)

// BenchmarkRerank は N=100,000 件に 768 次元のベクトルを付け、BM25 で M 件に絞ってから並べ直す時間を、全件のベクトル比較と IVF と比べる。
// 文書の語とベクトルは別々に作ったもので、互いに関係しない。時間だけを比べる。
//
//	go test -run '^$' -bench Rerank -benchtime 200x ./internal/search/
func BenchmarkRerank(b *testing.B) {
	const n, dim = 100_000, 768
	ds := randomDocs(n)
	ix := search.New(ds)
	flat, err := search.NewFlat(ds, clusteredVectors(n, dim, benchClusters, 1), nil)
	if err != nil {
		b.Fatal(err)
	}
	queries := clusteredVectors(benchQueries, dim, benchClusters, 2)
	b.Run("bm25", func(b *testing.B) {
		for b.Loop() {
			ix.Rank(benchQuery, benchTopK)
		}
		b.ReportMetric(float64(ix.Rank(benchQuery, benchTopK).Scored), "candidates")
	})
	for _, cfg := range []search.RerankConfig{
		{Depth: 100},
		{Depth: 1_000},
		{Depth: 0},
		{Depth: 0, Operator: search.OperatorAnd},
	} {
		r, err := search.NewRerank(ix, flat, cfg)
		if err != nil {
			b.Fatal(err)
		}
		op := "or"
		if cfg.Operator == search.OperatorAnd {
			op = "and"
		}
		b.Run(fmt.Sprintf("rerank/%s/M=%d", op, cfg.Depth), func(b *testing.B) {
			i := 0
			for b.Loop() {
				if _, err := r.RankVector(benchQuery, queries[i%len(queries)], benchTopK); err != nil {
					b.Fatal(err)
				}
				i++
			}
		})
	}
	b.Run("flat", func(b *testing.B) {
		i := 0
		for b.Loop() {
			if _, err := flat.SearchVector(queries[i%len(queries)], benchTopK); err != nil {
				b.Fatal(err)
			}
			i++
		}
	})
	ivf, err := search.NewIVF(ds, clusteredVectors(n, dim, benchClusters, 1), nil, search.IVFConfig{})
	if err != nil {
		b.Fatal(err)
	}
	for _, nprobe := range []int{1, 16} {
		approx := ivf.WithNProbe(nprobe)
		b.Run(fmt.Sprintf("ivf/nlist=%d/nprobe=%d", ivf.NList(), nprobe), func(b *testing.B) {
			i := 0
			for b.Loop() {
				if _, err := approx.SearchVector(queries[i%len(queries)], benchTopK); err != nil {
					b.Fatal(err)
				}
				i++
			}
		})
	}
}
