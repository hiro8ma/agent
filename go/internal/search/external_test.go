package search_test

import (
	"fmt"
	"math"
	"path/filepath"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/hiro8ma/agent/go/internal/search"
)

// TestBuildPostingsFileExternalMatchesMemory は、外部ソートで作ったファイルが、メモリ上の索引から WritePostingsFile で書いたものとバイト単位で一致することを、メモリの上限ごとに確かめる。
func TestBuildPostingsFileExternalMatchesMemory(t *testing.T) {
	t.Parallel()
	corpora := map[string][]search.Doc{
		"latin":    randomDocs(2_000),
		"japanese": randomJapaneseDocs(1_000),
	}
	testCases := map[string]struct {
		budget int
		// perDoc なら上限が 1 文書の出現記録より小さく、文書ごとに 1 つの並びになる。
		perDoc bool
	}{
		"上限 1 件で文書ごとに並びを書く": {budget: 1, perDoc: true},
		"上限 500 件":     {budget: 500},
		"上限 20,000 件":  {budget: 20_000},
		"上限なしで並びは 1 つ": {budget: math.MaxInt},
	}
	for cn, docs := range corpora {
		mem := writePostingsFile(t, search.New(docs))
		for tn, tc := range testCases {
			t.Run(cn+"/"+tn, func(t *testing.T) {
				t.Parallel()
				dir := t.TempDir()
				pf, stats, err := search.BuildPostingsFileExternal(docs, nil, tc.budget, dir, filepath.Join(dir, "postings.bin"))
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() { _ = pf.Close() })
				if diff := search.SamePostingsFile(mem, pf); diff != "" {
					t.Fatalf("メモリ上の索引と違う: %s", diff)
				}
				if tc.perDoc && stats.Runs != len(docs) {
					t.Errorf("Runs = %d, want %d", stats.Runs, len(docs))
				}
				if tc.budget == math.MaxInt && stats.Runs != 1 {
					t.Errorf("Runs = %d, want 1", stats.Runs)
				}
				if !tc.perDoc && stats.MaxBuffered > tc.budget {
					t.Errorf("MaxBuffered = %d, want at most %d", stats.MaxBuffered, tc.budget)
				}
				ids, err := pf.And("w00001", "w00002")
				if err != nil {
					t.Fatal(err)
				}
				want, err := mem.And("w00001", "w00002")
				if err != nil {
					t.Fatal(err)
				}
				if fmt.Sprint(ids) != fmt.Sprint(want) {
					t.Errorf("And = %v, want %v", ids, want)
				}
			})
		}
	}
}

// peakHeap は build を走らせる間、1ms ごとにヒープの使用量を見て、始める前からの最大の増分を返す。
func peakHeap(build func()) uint64 {
	runtime.GC()
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	base := ms.HeapAlloc
	var peak atomic.Uint64
	done := make(chan struct{})
	var wg sync.WaitGroup
	wg.Go(func() {
		tick := time.NewTicker(time.Millisecond)
		defer tick.Stop()
		for {
			var m runtime.MemStats
			runtime.ReadMemStats(&m)
			if m.HeapAlloc > base && m.HeapAlloc-base > peak.Load() {
				peak.Store(m.HeapAlloc - base)
			}
			select {
			case <-done:
				return
			case <-tick.C:
			}
		}
	})
	build()
	close(done)
	wg.Wait()
	return peak.Load()
}

// BenchmarkBuildPostingsExternal は 10 万文書で、メモリ上の索引を作って書き出す方法と、外部ソートでメモリの上限を変えた場合の、作る時間とヒープの最大の増分を比べる。
//
//	go test -run '^$' -bench BuildPostingsExternal -benchtime 3x ./internal/search/
func BenchmarkBuildPostingsExternal(b *testing.B) {
	docs := randomDocs(100_000)
	b.Run("memory", func(b *testing.B) {
		var peak uint64
		for b.Loop() {
			dir := b.TempDir()
			peak = max(peak, peakHeap(func() {
				pf, err := search.WritePostingsFile(search.New(docs), filepath.Join(dir, "postings.bin"))
				if err != nil {
					b.Fatal(err)
				}
				_ = pf.Close()
			}))
		}
		b.ReportMetric(float64(peak)/(1<<20), "peak-MB")
	})
	for _, budget := range []int{50_000, 500_000, math.MaxInt} {
		name := fmt.Sprintf("external/budget=%d", budget)
		if budget == math.MaxInt {
			name = "external/budget=inf"
		}
		b.Run(name, func(b *testing.B) {
			var (
				peak  uint64
				stats search.ExternalStats
			)
			for b.Loop() {
				dir := b.TempDir()
				peak = max(peak, peakHeap(func() {
					pf, s, err := search.BuildPostingsFileExternal(docs, nil, budget, dir, filepath.Join(dir, "postings.bin"))
					if err != nil {
						b.Fatal(err)
					}
					stats = s
					_ = pf.Close()
				}))
			}
			b.ReportMetric(float64(peak)/(1<<20), "peak-MB")
			b.ReportMetric(float64(stats.Runs), "runs")
			b.ReportMetric(float64(stats.MaxBuffered), "buffered")
		})
	}
}
