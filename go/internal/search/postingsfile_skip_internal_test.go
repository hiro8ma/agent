package search

import (
	"cmp"
	"fmt"
	"math/rand/v2"
	"path/filepath"
	"slices"
	"testing"
	"time"
)

func writeLists(tb testing.TB, lists map[string][]int32) *PostingsFile {
	tb.Helper()
	words := make([]string, 0, len(lists))
	for w := range lists {
		words = append(words, w)
	}
	slices.Sort(words)
	postings := make([][]posting, len(words))
	for i, w := range words {
		postings[i] = postingsFor(lists[w])
	}
	pf, err := writePostings(words, postings, filepath.Join(tb.TempDir(), "postings.bin"))
	if err != nil {
		tb.Fatal(err)
	}
	tb.Cleanup(func() { _ = pf.Close() })
	return pf
}

type diskAndMethod struct {
	name string
	skip *SkipConfig
}

var diskAndMethods = []diskAndMethod{
	{name: "順に読む"},
	{name: "√n 二分探索", skip: &SkipConfig{}},
	{name: "64 件 二分探索", skip: &SkipConfig{Interval: 64}},
	{name: "16 件 1 段", skip: &SkipConfig{Interval: 16, Linear: true}},
	{name: "16 件 2 段(×16)", skip: &SkipConfig{Interval: 16, Long: 16}},
}

func runDiskAnd(t *testing.T, pf *PostingsFile, m diskAndMethod, terms []string) ([]int, cursorStats, time.Duration) {
	t.Helper()
	pf.skips = nil
	if m.skip != nil {
		if _, err := pf.BuildSkipIndex(*m.skip); err != nil {
			t.Fatal(err)
		}
	}
	var st cursorStats
	got, err := pf.and(terms, &st)
	if err != nil {
		t.Fatal(err)
	}
	const reps = 20
	start := time.Now()
	for range reps {
		if _, err := pf.and(terms, &cursorStats{}); err != nil {
			t.Fatal(err)
		}
	}
	return got, st, time.Since(start) / reps
}

func loadedStats(pf *PostingsFile, terms []string) cursorStats {
	var st cursorStats
	spans, _ := pf.spans(terms)
	for _, s := range spans {
		st.entries += int64(s.count)
		st.decoded += s.size
		st.fileBytes += s.size
	}
	return st
}

func TestPostingsFileSkipMatchesSequential(t *testing.T) {
	t.Parallel()
	pf, err := WritePostingsFile(New(zipfDocs(5_000)), filepath.Join(t.TempDir(), "postings.bin"))
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = pf.Close() }()
	r := rand.New(rand.NewPCG(1, 2))
	var queries [][]string
	for range 200 {
		terms := make([]string, 2+r.IntN(3))
		for i := range terms {
			terms[i] = fmt.Sprintf("w%05d", 1+r.IntN(500))
		}
		queries = append(queries, terms)
	}
	want := make([][]int, len(queries))
	for i, q := range queries {
		if want[i], err = pf.And(q...); err != nil {
			t.Fatal(err)
		}
	}
	for _, m := range diskAndMethods[1:] {
		if _, err := pf.BuildSkipIndex(*m.skip); err != nil {
			t.Fatal(err)
		}
		for i, q := range queries {
			got, err := pf.And(q...)
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(got, want[i]) {
				t.Fatalf("%s %v: got %v, want %v", m.name, q, got, want[i])
			}
		}
	}
}

// TestPostingsFileSkipReadsLess は長さの比 100 の AND で、飛び先を使うと読む出現記録の件数が順に読むときの半分未満になることを確かめる。
func TestPostingsFileSkipReadsLess(t *testing.T) {
	t.Parallel()
	r := rand.New(rand.NewPCG(4, 4))
	pf := writeLists(t, map[string][]int32{"s": sortedIDs(r, 100, 20_000), "l": sortedIDs(r, 10_000, 20_000)})
	want, seq, _ := runDiskAnd(t, pf, diskAndMethods[0], []string{"s", "l"})
	got, skip, _ := runDiskAnd(t, pf, diskAndMethods[1], []string{"s", "l"})
	if !slices.Equal(got, want) {
		t.Fatalf("got %d ids, want %d", len(got), len(want))
	}
	if skip.entries*2 >= seq.entries {
		t.Fatalf("読んだ出現記録 = 飛び先あり %d / なし %d", skip.entries, seq.entries)
	}
}

// TestPostingsFileSkipRatios は短い語を 1,000 件に固定し、長さの比ごとに、順に読む方法、飛び先の作り方ごと、全部を展開してから galloping で引く方法で、
// 読んだ出現記録の件数とバイト数、ファイルから読んだバイト数、時間を出し、結果が一致することを確かめる。
//
//	go test -run TestPostingsFileSkipRatios -v ./internal/search/
func TestPostingsFileSkipRatios(t *testing.T) {
	t.Parallel()
	const short = 1_000
	for _, ratio := range []int{1, 10, 100, 1_000} {
		long := short * ratio
		r := rand.New(rand.NewPCG(uint64(ratio), 9))
		pf := writeLists(t, map[string][]int32{"s": sortedIDs(r, short, long*2), "l": sortedIDs(r, long, long*2)})
		terms := []string{"s", "l"}
		var want []int
		for i, m := range diskAndMethods {
			got, st, d := runDiskAnd(t, pf, m, terms)
			if i == 0 {
				want = got
			} else if !slices.Equal(got, want) {
				t.Fatalf("ratio=%d %s: got %d ids, want %d", ratio, m.name, len(got), len(want))
			}
			t.Logf("比 %4d %-16s 件数 %8d / 復号 %8d B / ファイル %8d B / 飛び先の比較 %6d / %8.1fµs", ratio, m.name, st.entries, st.decoded, st.fileBytes, st.skipCompares, float64(d.Nanoseconds())/1e3)
		}
		pf.skips = nil
		got, err := pf.AndLoaded(terms...)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(got, want) {
			t.Fatalf("ratio=%d loaded: got %d ids, want %d", ratio, len(got), len(want))
		}
		const reps = 20
		start := time.Now()
		for range reps {
			if _, err := pf.AndLoaded(terms...); err != nil {
				t.Fatal(err)
			}
		}
		st := loadedStats(pf, terms)
		t.Logf("比 %4d %-16s 件数 %8d / 復号 %8d B / ファイル %8d B / %8.1fµs 共通 %d", ratio, "全部展開+二分探索", st.entries, st.decoded, st.fileBytes, float64(time.Since(start).Nanoseconds())/1e3/reps, len(want))
	}
}

// clusteredIDs は n 個の文書番号を、[0, universe) の中の clusters 個の区間に半分の密度で詰めて並べる。区間の位置は種で決める。
func clusteredIDs(r *rand.Rand, n, clusters, universe int) []int32 {
	per := n / clusters
	width := per * 2
	starts := sortedIDs(r, clusters, universe/width)
	var ids []int32
	for _, s := range starts {
		base := int(s) * width
		for _, off := range sortedIDs(r, per, width) {
			ids = append(ids, int32(base)+off)
		}
	}
	return ids
}

// TestClusteredDocIDs は同じ 1,000 件の語を、5 つの区間に固めた場合と全体に散らした場合で、5 万件の語との AND の比較の回数と読む件数を比べる。
//
//	go test -run TestClusteredDocIDs -v ./internal/search/
func TestClusteredDocIDs(t *testing.T) {
	t.Parallel()
	const (
		universe = 200_000
		rare     = 1_000
		frequent = 50_000
	)
	r := rand.New(rand.NewPCG(8, 8))
	long := sortedIDs(r, frequent, universe)
	ps := postingsFor(long)
	sp := newSkipPostings(ps, sqrtInterval(len(ps)))
	for _, layout := range []struct {
		name string
		ids  []int32
	}{
		{name: "区間に固める", ids: clusteredIDs(r, rare, 5, universe)},
		{name: "全体に散らす", ids: sortedIDs(r, rare, universe)},
	} {
		want := naiveIntersectSorted(layout.ids, ps)
		merged, nMerge := countMerge(slices.Clone(layout.ids), ps)
		galloped, nGallop := countGallop(slices.Clone(layout.ids), ps)
		skipped, nSkip := intersectSkip(slices.Clone(layout.ids), sp, selectFields(nil), true)
		for name, got := range map[string][]int32{"merge": merged, "gallop": galloped, "skip": skipped} {
			if !slices.Equal(got, want) {
				t.Fatalf("%s %s: got %d ids, want %d", layout.name, name, len(got), len(want))
			}
		}
		pf := writeLists(t, map[string][]int32{"rare": layout.ids, "frequent": long})
		terms := []string{"rare", "frequent"}
		_, seq, dSeq := runDiskAnd(t, pf, diskAndMethods[0], terms)
		got, skip, dSkip := runDiskAnd(t, pf, diskAndMethods[1], terms)
		if !slices.Equal(got, int32sToInts(want)) {
			t.Fatalf("%s disk: got %d ids, want %d", layout.name, len(got), len(want))
		}
		t.Logf("%s: 比較 merge %6d / gallop %6d / skip(√n) %6d。ファイル 順に読む %6d 件 %7.1fµs / 飛び先 %6d 件 %7.1fµs。共通 %d",
			layout.name, nMerge, nGallop, nSkip, seq.entries, float64(dSeq.Nanoseconds())/1e3, skip.entries, float64(dSkip.Nanoseconds())/1e3, len(want))
	}
}

func int32sToInts(ids []int32) []int {
	out := make([]int, len(ids))
	for i, id := range ids {
		out[i] = int(id)
	}
	return out
}

// TestAndOrderIntermediateSizes は 4 語の AND を、短い postings から畳み込む順（intersectAll）と長い方から畳み込む順で、途中の結果の件数と比較の回数を比べる。結果は一致する。
//
//	go test -run TestAndOrderIntermediateSizes -v ./internal/search/
func TestAndOrderIntermediateSizes(t *testing.T) {
	t.Parallel()
	ix := New(zipfDocs(100_000))
	all := selectFields(nil)
	r := rand.New(rand.NewPCG(6, 6))
	var rareTotal, freqTotal [3]int
	var rareCmp, freqCmp int
	var rareTime, freqTime time.Duration
	var lengths [4]int
	const queries = 50
	for range queries {
		var lists [][]posting
		var terms []string
		for len(lists) < 4 {
			term := fmt.Sprintf("w%05d", 1+r.IntN(300))
			if slices.Contains(terms, term) {
				continue
			}
			terms = append(terms, term)
			lists = append(lists, ix.postingsOf(ix.lookupTerm(term)))
		}
		var steps []int
		got := intersectAll(slices.Clone(lists), all, true, defaultGallopRatio, func(_ int, ids []int32) { steps = append(steps, len(ids)) })
		asc := slices.Clone(lists)
		slices.SortFunc(asc, func(a, b []posting) int { return cmp.Compare(len(a), len(b)) })
		for i, ps := range asc {
			lengths[i] += len(ps)
		}
		desc := slices.Clone(asc)
		slices.Reverse(desc)
		rare, rareSizes, nRare, dRare := foldAnd(asc)
		freq, freqSizes, nFreq, dFreq := foldAnd(desc)
		if !slices.Equal(got, freq) || !slices.Equal(rare, freq) || !slices.Equal(steps[1:], rareSizes[:len(steps)-1]) {
			t.Fatalf("%v: intersectAll %d ids / まれな語から %d / 多い語から %d", terms, len(got), len(rare), len(freq))
		}
		rareCmp, freqCmp = rareCmp+nRare, freqCmp+nFreq
		rareTime, freqTime = rareTime+dRare, freqTime+dFreq
		for i := range 3 {
			rareTotal[i] += rareSizes[i]
			freqTotal[i] += freqSizes[i]
		}
	}
	t.Logf("postings の長さの平均（短い順）: %d / %d / %d / %d", lengths[0]/queries, lengths[1]/queries, lengths[2]/queries, lengths[3]/queries)
	t.Logf("途中の結果の平均（1 回目 / 2 回目 / 3 回目の畳み込みの後）: まれな語から %.0f / %.0f / %.0f、多い語から %.0f / %.0f / %.0f",
		float64(rareTotal[0])/queries, float64(rareTotal[1])/queries, float64(rareTotal[2])/queries,
		float64(freqTotal[0])/queries, float64(freqTotal[1])/queries, float64(freqTotal[2])/queries)
	t.Logf("比較の回数と時間の平均: まれな語から %.0f 回 %.1fµs、多い語から %.0f 回 %.1fµs",
		float64(rareCmp)/queries, float64(rareTime.Nanoseconds())/1e3/queries, float64(freqCmp)/queries, float64(freqTime.Nanoseconds())/1e3/queries)
}

// foldAnd は lists をこの順に、intersect と同じく長さの比で 2 つのポインタと galloping を切り替えて畳み込む。途中で空になっても最後まで畳み込み、各回の後の件数と比較の回数と時間を返す。
func foldAnd(lists [][]posting) ([]int32, []int, int, time.Duration) {
	all := selectFields(nil)
	start := time.Now()
	ids := docsOf(lists[0], all, true)
	for _, ps := range lists[1:] {
		ids = intersect(ids, ps, all, true, defaultGallopRatio)
	}
	d := time.Since(start)
	ids2 := docsOf(lists[0], all, true)
	var sizes []int
	n := 0
	for _, ps := range lists[1:] {
		var c int
		if useGallop(len(ids2), len(ps), defaultGallopRatio) {
			ids2, c = countGallop(ids2, ps)
		} else {
			ids2, c = countMerge(ids2, ps)
		}
		n += c
		sizes = append(sizes, len(ids2))
	}
	return ids, sizes, n, d
}
