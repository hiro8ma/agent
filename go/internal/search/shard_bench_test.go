package search_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"

	"github.com/hiro8ma/agent/go/internal/search"
)

var shardCounts = []int{1, 2, 5, 20}

// shardQueries は benchQuery に、Zipf の順位 1 から 5,000 の語を 2 つ並べたクエリを足す。fixed を渡すと 1 語目をその語に固定する。
func shardQueries(n int, fixed string) []string {
	r := rand.New(rand.NewPCG(3, 5))
	qs := []string{benchQuery}
	if fixed != "" {
		qs = nil
	}
	for len(qs) < n {
		a := fmt.Sprintf("w%05d", 1+r.IntN(5_000))
		if fixed != "" {
			a = fixed
		}
		qs = append(qs, fmt.Sprintf("%s w%05d", a, 1+r.IntN(5_000)))
	}
	return qs
}

func overlapAt(want, got []search.Hit) float64 {
	if len(want) == 0 {
		return 1
	}
	ids := make(map[string]struct{}, len(got))
	for _, h := range got {
		ids[h.Doc.ID] = struct{}{}
	}
	n := 0
	for _, h := range want {
		if _, ok := ids[h.Doc.ID]; ok {
			n++
		}
	}
	return float64(n) / float64(len(want))
}

// kendallTau は 1 台の索引の上位の文書について、1 台の点数とシャードでの点数の順位の一致を τ-b で返す。
func kendallTau(want []search.Hit, got map[string]float64) float64 {
	var conc, disc, tieX, tieY float64
	for i := range want {
		for j := i + 1; j < len(want); j++ {
			x := want[i].Score - want[j].Score
			y := got[want[i].Doc.ID] - got[want[j].Doc.ID]
			switch {
			case x == 0 && y == 0:
			case x == 0:
				tieX++
			case y == 0:
				tieY++
			case (x > 0) == (y > 0):
				conc++
			default:
				disc++
			}
		}
	}
	d := (conc + disc + tieX) * (conc + disc + tieY)
	if d == 0 {
		return 1
	}
	return (conc - disc) / math.Sqrt(d)
}

type agreement struct{ overlap, tau float64 }

func agree(single *search.Index, sh *search.Sharded, queries []string) agreement {
	var a agreement
	for _, q := range queries {
		want := single.Rank(q, 10).Hits
		all := sh.Rank(q, -1).Hits
		scores := make(map[string]float64, len(all))
		for _, h := range all {
			scores[h.Doc.ID] = h.Score
		}
		a.overlap += overlapAt(want, sh.Rank(q, 10).Hits)
		a.tau += kendallTau(want, scores)
	}
	a.overlap /= float64(len(queries))
	a.tau /= float64(len(queries))
	return a
}

func benchShardedTypes(b *testing.B, name string, single *search.Index, sh *search.Sharded, queries []string) {
	for _, typ := range []struct {
		name string
		t    search.SearchType
	}{{name: "query_then_fetch", t: search.QueryThenFetch}, {name: "dfs_query_then_fetch", t: search.DFSQueryThenFetch}} {
		s := *sh
		s.Type = typ.t
		a := agree(single, &s, queries)
		b.Run(name+"/"+typ.name, func(b *testing.B) {
			var stats int64
			for b.Loop() {
				stats += s.Rank(benchQuery, 10).StatsTime.Nanoseconds()
			}
			b.ReportMetric(float64(stats)/float64(b.N), "stats-ns/op")
			b.ReportMetric(a.overlap, "overlap@10")
			b.ReportMetric(a.tau, "tau")
		})
	}
}

func BenchmarkSharded(b *testing.B) {
	queries := shardQueries(50, "")
	for _, n := range benchSizes {
		ds := withIDs(randomDocs(n))
		single := search.New(ds)
		b.Run(fmt.Sprintf("N=%d/single", n), func(b *testing.B) {
			for b.Loop() {
				single.Rank(benchQuery, 10)
			}
		})
		for _, k := range shardCounts {
			benchShardedTypes(b, fmt.Sprintf("N=%d/shards=%d", n, k), single, search.NewSharded(ds, k, nil), queries)
		}
	}
}

// BenchmarkShardedSkew は、ある語を含む文書だけを 1 つのシャードに集めるカスタムルーティングで、その語を含むクエリの一致を測る。
func BenchmarkShardedSkew(b *testing.B) {
	const n, shards, term = 10_000, 5, "w00100"
	ds := withIDs(randomDocs(n))
	single := search.New(ds)
	hash := search.HashRouter(shards - 1)
	routers := map[string]search.Router{
		"hash": nil,
		"term-to-shard0": func(d search.Doc) int {
			if slices.Contains(strings.Fields(d.Content), term) {
				return 0
			}
			return 1 + hash(d)
		},
	}
	queries := shardQueries(50, term)
	for _, name := range []string{"hash", "term-to-shard0"} {
		sh := search.NewSharded(ds, shards, routers[name])
		b.Logf("%s: docs per shard = %v", name, sh.ShardDocs())
		benchShardedTypes(b, "N=10000/shards=5/"+name, single, sh, queries)
	}
}
