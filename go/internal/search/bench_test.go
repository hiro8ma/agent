package search_test

import (
	"context"
	"fmt"
	"math/rand/v2"
	"runtime"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/hiro8ma/agent/go/internal/search"
)

const benchQuery = "w00050 w02000"

var benchSizes = []int{1_000, 10_000, 100_000}

// randomDocs は語の出現頻度が Zipf 分布に従う文書を、固定の種で作る。語は固定幅なので部分一致で別の語に当たらない。
func randomDocs(n int) []search.Doc {
	r := rand.New(rand.NewPCG(42, uint64(n)))
	zipf := rand.NewZipf(r, 1.1, 1, 19_999)
	ds := make([]search.Doc, n)
	for i := range ds {
		words := make([]string, 20+r.IntN(61))
		for j := range words {
			words[j] = fmt.Sprintf("w%05d", zipf.Uint64())
		}
		ds[i] = search.Doc{Title: fmt.Sprintf("doc%d", i), Content: strings.Join(words, " ")}
	}
	return ds
}

type fullScan struct{ docs []search.Doc }

func (s fullScan) Search(_ context.Context, query string, limit int) ([]search.Doc, error) {
	terms := strings.Fields(query)
	type scored struct {
		doc   search.Doc
		score int
	}
	var results []scored
	for _, d := range s.docs {
		text := d.Title + " " + d.Content
		score := 0
		for _, t := range terms {
			if strings.Contains(text, t) {
				score++
			}
		}
		if score > 0 {
			results = append(results, scored{doc: d, score: score})
		}
	}
	sort.SliceStable(results, func(i, j int) bool { return results[i].score > results[j].score })
	if len(results) > limit {
		results = results[:limit]
	}
	out := make([]search.Doc, len(results))
	for i, r := range results {
		out[i] = r.doc
	}
	return out, nil
}

type searcher interface {
	Search(ctx context.Context, query string, limit int) ([]search.Doc, error)
}

func BenchmarkSearch(b *testing.B) {
	for _, n := range benchSizes {
		ds := randomDocs(n)
		ix := search.New(ds)
		searchers := []struct {
			name string
			s    searcher
		}{
			{name: "scan", s: fullScan{docs: ds}},
			{name: "bm25", s: ix},
		}
		for _, sc := range searchers {
			b.Run(fmt.Sprintf("N=%d/%s", n, sc.name), func(b *testing.B) {
				ctx := b.Context()
				for b.Loop() {
					if _, err := sc.s.Search(ctx, benchQuery, 10); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
		b.Run(fmt.Sprintf("N=%d/candidates", n), func(b *testing.B) {
			var scored int
			for b.Loop() {
				scored = ix.Rank(benchQuery, 10).Scored
			}
			b.ReportMetric(float64(scored), "scored/op")
			b.ReportMetric(float64(scored)/float64(n)*100, "%docs")
		})
	}
}

func BenchmarkBuild(b *testing.B) {
	for _, n := range benchSizes {
		ds := randomDocs(n)
		b.Run(fmt.Sprintf("N=%d", n), func(b *testing.B) {
			for b.Loop() {
				search.New(ds)
			}
			b.ReportMetric(float64(retainedHeap(func() any { return search.New(ds) })), "heap-B")
		})
	}
}

// retainedHeap は build が返したものを保持したまま GC したときに増えたヒープの大きさを測る。
func retainedHeap(build func() any) uint64 {
	var before, after runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&before)
	v := build()
	runtime.GC()
	runtime.ReadMemStats(&after)
	runtime.KeepAlive(v)
	return after.HeapAlloc - before.HeapAlloc
}

func BenchmarkStages(b *testing.B) {
	for _, n := range benchSizes {
		ix := search.New(randomDocs(n))
		b.Run(fmt.Sprintf("N=%d", n), func(b *testing.B) {
			var match, rank time.Duration
			for b.Loop() {
				res := ix.Rank(benchQuery, 10)
				match += res.MatchTime
				rank += res.RankTime
			}
			ops := float64(b.N)
			b.ReportMetric(float64(match.Nanoseconds())/ops, "match-ns/op")
			b.ReportMetric(float64(rank.Nanoseconds())/ops, "rank-ns/op")
			b.ReportMetric(float64(match)/float64(match+rank)*100, "match-%")
		})
	}
}

var particles = []string{"の", "は", "が", "を", "に", "で", "と", "も", "には", "では", "への"}

// randomJapaneseDocs は 2 文字の漢字語を Zipf 分布で選び、助詞でつないだ文書を作る。
func randomJapaneseDocs(n int) []search.Doc {
	r := rand.New(rand.NewPCG(7, uint64(n)))
	zipf := rand.NewZipf(r, 1.1, 1, 19_999)
	ds := make([]search.Doc, n)
	for i := range ds {
		var sb strings.Builder
		for range 10 + r.IntN(31) {
			sb.WriteString(kanjiWord(int(zipf.Uint64())))
			sb.WriteString(particles[r.IntN(len(particles))])
		}
		ds[i] = search.Doc{Title: fmt.Sprintf("doc%d", i), Content: sb.String()}
	}
	return ds
}

func kanjiWord(i int) string {
	return string([]rune{rune(0x4E00 + i/200), rune(0x4F00 + i%200)})
}

func BenchmarkStopWords(b *testing.B) {
	corpora := []struct {
		name  string
		docs  func(int) []search.Doc
		query string
	}{
		{name: "latin", docs: randomDocs, query: benchQuery},
		{name: "japanese", docs: randomJapaneseDocs, query: kanjiWord(50) + "には" + kanjiWord(2000)},
	}
	analyzers := []struct {
		name string
		a    *search.Analyzer
	}{
		{name: "keep", a: search.NewAnalyzer()},
		{name: "stop", a: search.NewAnalyzer(search.WithStopWords(search.DefaultStopWords()...))},
	}
	for _, c := range corpora {
		for _, n := range benchSizes {
			ds := c.docs(n)
			for _, an := range analyzers {
				ix := search.New(ds, search.WithAnalyzer(an.a))
				st := ix.Stats()
				b.Run(fmt.Sprintf("%s/N=%d/%s", c.name, n, an.name), func(b *testing.B) {
					var scored int
					for b.Loop() {
						scored = ix.Rank(c.query, 10).Scored
					}
					b.ReportMetric(float64(st.Terms), "terms")
					b.ReportMetric(float64(st.Postings), "postings")
					b.ReportMetric(float64(scored), "scored/op")
				})
			}
		}
	}
}
