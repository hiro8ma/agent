package search_test

import (
	"fmt"
	"testing"

	"github.com/hiro8ma/agent/go/internal/search"
)

var expansionSizes = []int{0, 5, 20, 50, 200}

const expansionGroups = 50

// benchThesaurus は語 1 つあたり n 語を足すグループを 50 個作る。benchQuery の 2 語を先頭の 2 グループの見出しにし、残りの語は w03000 から順に重ならないように割り当てる。
func benchThesaurus(n int) *search.Thesaurus {
	heads := []int{50, 2000}
	next := 3000
	opts := []search.ThesaurusOption{search.WithExpansionLimit(-1)}
	for g := range expansionGroups {
		var group []string
		if g < len(heads) {
			group = append(group, fmt.Sprintf("w%05d", heads[g]))
		}
		for len(group) < n+1 {
			group = append(group, fmt.Sprintf("w%05d", next))
			next++
		}
		opts = append(opts, search.WithSynonymGroup(group...))
	}
	return search.NewThesaurus(opts...)
}

// BenchmarkExpansionQuery は語 1 つあたりに足す語の数ごとに、クエリ拡張と文書拡張の検索時間を比べる。agree@10 は両者の上位 10 件の一致率。
func BenchmarkExpansionQuery(b *testing.B) {
	ds := randomDocs(10_000)
	plain := search.New(ds)
	for _, n := range expansionSizes {
		th := benchThesaurus(n)
		expanded := search.New(ds, search.WithDocExpansion(th))
		byQuery := plain.RankQuery(search.Query{Text: benchQuery, Thesaurus: th}, 10)
		byDoc := expanded.RankQuery(search.Query{Text: benchQuery}, 10)
		agree := overlap(byQuery.Hits, byDoc.Hits)
		b.Run(fmt.Sprintf("n=%d/query", n), func(b *testing.B) {
			var res search.Result
			for b.Loop() {
				res = plain.RankQuery(search.Query{Text: benchQuery, Thesaurus: th}, 10)
			}
			b.ReportMetric(float64(res.Scored), "scored/op")
			b.ReportMetric(agree, "agree@10")
		})
		b.Run(fmt.Sprintf("n=%d/doc", n), func(b *testing.B) {
			var res search.Result
			for b.Loop() {
				res = expanded.RankQuery(search.Query{Text: benchQuery}, 10)
			}
			b.ReportMetric(float64(res.Scored), "scored/op")
			b.ReportMetric(agree, "agree@10")
		})
	}
}

// BenchmarkExpansionBuild は文書拡張の有無と足す語の数ごとに、索引を作る時間と大きさを比べる。
func BenchmarkExpansionBuild(b *testing.B) {
	ds := randomDocs(10_000)
	for _, n := range expansionSizes {
		var opts []search.Option
		if n > 0 {
			opts = append(opts, search.WithDocExpansion(benchThesaurus(n)))
		}
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			var ix *search.Index
			for b.Loop() {
				ix = search.New(ds, opts...)
			}
			st := ix.Stats()
			structs, positions := search.PostingsBytes(ix)
			b.ReportMetric(float64(st.Positions), "positions")
			b.ReportMetric(float64(st.AddedPositions), "added")
			b.ReportMetric(float64(structs+positions), "postings-B")
		})
	}
}
