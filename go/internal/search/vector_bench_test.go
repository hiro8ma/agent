package search_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
	"time"

	"github.com/hiro8ma/agent/go/internal/search"
)

type vectorSearcher interface {
	SearchVector(q []float32, limit int) ([]search.VectorHit, error)
}

// clusteredVectors は clusters 個の中心のまわりに点をばらまく。中心の成分は標準正規分布、ずれは標準偏差 0.5。
// 中心は seed によらず clusters と dim だけで決まるので、同じ引数の文書とクエリは同じ塊を共有する。
func clusteredVectors(n, dim, clusters int, seed uint64) [][]float32 {
	cr := rand.New(rand.NewPCG(uint64(dim), uint64(clusters)))
	centers := make([][]float32, clusters)
	for c := range centers {
		centers[c] = make([]float32, dim)
		for j := range centers[c] {
			centers[c][j] = float32(cr.NormFloat64())
		}
	}
	r := rand.New(rand.NewPCG(seed, uint64(n)))
	vecs := make([][]float32, n)
	for i := range vecs {
		c := centers[r.IntN(clusters)]
		v := make([]float32, dim)
		for j := range v {
			v[j] = c[j] + float32(0.5*r.NormFloat64())
		}
		vecs[i] = v
	}
	return vecs
}

// uniformVectors は球面上に一様な向きの点を作る。近い点が固まらないので、IVF にとって最も不利な分布になる。
func uniformVectors(n, dim int, seed uint64) [][]float32 {
	r := rand.New(rand.NewPCG(seed, uint64(n)))
	vecs := make([][]float32, n)
	for i := range vecs {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(r.NormFloat64())
		}
		vecs[i] = v
	}
	return vecs
}

// meanRecall は exact の上位 k 件のうち approx の上位 k 件にも入った割合をクエリで平均する。
func meanRecall(tb testing.TB, exact, approx vectorSearcher, queries [][]float32, k int) float64 {
	tb.Helper()
	truth := make([][]search.VectorHit, len(queries))
	for i, q := range queries {
		hits, err := exact.SearchVector(q, k)
		if err != nil {
			tb.Fatalf("exact: %v", err)
		}
		truth[i] = hits
	}
	return recallAgainst(tb, truth, approx, queries, k)
}

func recallAgainst(tb testing.TB, truth [][]search.VectorHit, approx vectorSearcher, queries [][]float32, k int) float64 {
	tb.Helper()
	sum := 0.0
	for i, q := range queries {
		hits, err := approx.SearchVector(q, k)
		if err != nil {
			tb.Fatalf("approx: %v", err)
		}
		want := make(map[int]struct{}, len(truth[i]))
		for _, h := range truth[i] {
			want[h.ID] = struct{}{}
		}
		found := 0
		for _, h := range hits {
			if _, ok := want[h.ID]; ok {
				found++
			}
		}
		sum += float64(found) / float64(len(truth[i]))
	}
	return sum / float64(len(queries))
}

const (
	benchQueries  = 100
	benchClusters = 100
	benchTopK     = 10
)

// BenchmarkVectorSearch は nlist と nprobe ごとに recall@10、1 回の検索時間、索引の構築時間を出す。
//
//	go test -run '^$' -bench VectorSearch -benchtime 200x ./internal/search/
func BenchmarkVectorSearch(b *testing.B) {
	for _, dist := range []string{"clustered", "uniform"} {
		for _, dim := range []int{384, 768} {
			for _, n := range []int{10_000, 100_000} {
				b.Run(fmt.Sprintf("%s/dim=%d/N=%d", dist, dim, n), func(b *testing.B) { benchVectorSearch(b, dist, n, dim) })
			}
		}
	}
}

func benchVectorSearch(b *testing.B, dist string, n, dim int) {
	var docs, queries [][]float32
	if dist == "clustered" {
		docs = clusteredVectors(n, dim, benchClusters, 1)
		queries = clusteredVectors(benchQueries, dim, benchClusters, 2)
	} else {
		docs = uniformVectors(n, dim, 1)
		queries = uniformVectors(benchQueries, dim, 2)
	}
	flat, err := search.NewFlat(make([]search.Doc, n), docs, nil)
	if err != nil {
		b.Fatal(err)
	}
	truth := make([][]search.VectorHit, len(queries))
	for i, q := range queries {
		if truth[i], err = flat.SearchVector(q, benchTopK); err != nil {
			b.Fatal(err)
		}
	}
	runQueries(b, "flat", flat, queries, 1, 0)

	sqrtN := int(math.Round(math.Sqrt(float64(n))))
	for _, nlist := range []int{sqrtN, 4 * sqrtN} {
		b.Run(fmt.Sprintf("ivf/nlist=%d", nlist), func(b *testing.B) { benchIVF(b, docs, queries, truth, nlist) })
	}
}

func benchIVF(b *testing.B, docs, queries [][]float32, truth [][]search.VectorHit, nlist int) {
	start := time.Now()
	ivf, err := search.NewIVF(make([]search.Doc, len(docs)), docs, nil, search.IVFConfig{NList: nlist})
	if err != nil {
		b.Fatal(err)
	}
	build := time.Since(start)
	for _, nprobe := range []int{1, 4, 16, 64} {
		approx := ivf.WithNProbe(nprobe)
		recall := recallAgainst(b, truth, approx, queries, benchTopK)
		runQueries(b, fmt.Sprintf("nprobe=%d", nprobe), approx, queries, recall, build)
	}
}

func runQueries(b *testing.B, name string, s vectorSearcher, queries [][]float32, recall float64, build time.Duration) {
	b.Run(name, func(b *testing.B) {
		i := 0
		for b.Loop() {
			if _, err := s.SearchVector(queries[i%len(queries)], benchTopK); err != nil {
				b.Fatal(err)
			}
			i++
		}
		b.ReportMetric(recall, "recall@10")
		b.ReportMetric(build.Seconds(), "build-s")
	})
}
