package search_test

import (
	"fmt"
	"path/filepath"
	"slices"
	"testing"

	"github.com/hiro8ma/agent/go/internal/search"
)

func writePostingsFile(tb testing.TB, ix *search.Index) *search.PostingsFile {
	tb.Helper()
	pf, err := search.WritePostingsFile(ix, filepath.Join(tb.TempDir(), "postings.bin"))
	if err != nil {
		tb.Fatal(err)
	}
	tb.Cleanup(func() { _ = pf.Close() })
	return pf
}

func TestPostingsFileRoundTrip(t *testing.T) {
	t.Parallel()
	ix := whitespaceIndex(tutorialIndexDocs...)
	pf := writePostingsFile(t, ix)
	testCases := map[string]struct {
		term string
	}{
		"3 文書すべてに出る語":   {term: "カレー"},
		"文書 1 と 3 に出る語": {term: "ライス"},
		"文書 3 だけに出る語":   {term: "カツ"},
		"語彙に無い語は空":      {term: "ナン"},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			want := []search.FilePosting{}
			for _, p := range search.PostingsOf(ix, tc.term) {
				want = append(want, search.FilePosting{Doc: p.Doc, TF: p.TF})
			}
			got, err := pf.Postings(tc.term)
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(got, want) {
				t.Fatalf("postings = %v, want %v", got, want)
			}
		})
	}
}

// TestPostingsFileDeltaSize は文書番号を差分で書くことを大きさで確かめる。b は文書 0 から 199、a は 200 から 202 に出る。
// 差分なら b は 1 バイトの差分と出現回数が 200 件で 400 バイト、a は 200 が 2 バイト、続く差分 1 が 1 バイトずつ、出現回数が 3 バイトで 7 バイト。
func TestPostingsFileDeltaSize(t *testing.T) {
	t.Parallel()
	contents := make([]string, 203)
	for i := range contents {
		contents[i] = "b"
		if i >= 200 {
			contents[i] = "a"
		}
	}
	pf := writePostingsFile(t, whitespaceIndex(contents...))
	if got := pf.Size(); got != 407 {
		t.Fatalf("Size = %d, want 407", got)
	}
	got, err := pf.Postings("a")
	if err != nil {
		t.Fatal(err)
	}
	if want := []search.FilePosting{{Doc: 200, TF: 1}, {Doc: 201, TF: 1}, {Doc: 202, TF: 1}}; !slices.Equal(got, want) {
		t.Fatalf("postings = %v, want %v", got, want)
	}
}

func TestPostingsFileAndMatchesMemory(t *testing.T) {
	t.Parallel()
	tutorial := whitespaceIndex(tutorialIndexDocs...)
	random := search.New(randomDocs(3_000))
	testCases := map[string]struct {
		ix   *search.Index
		text string
	}{
		"教材のカレー AND ライス":         {ix: tutorial, text: "カレー ライス"},
		"教材のカツ AND 合う AND ライス":   {ix: tutorial, text: "カツ 合う ライス"},
		"教材の共通部分が空になる語の組":        {ix: tutorial, text: "スパイス ライス"},
		"語彙に無い語を含めば空":            {ix: tutorial, text: "カレー ナン"},
		"合成データのまれな語の組":           {ix: random, text: benchQuery},
		"合成データのよく出る 3 語":         {ix: random, text: "w00001 w00002 w00003"},
		"合成データのよく出る語とまれな語":       {ix: random, text: "w00001 w00300"},
		"合成データで同じ語を 2 回書いても同じ結果": {ix: random, text: "w00002 w00002 w00010"},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			pf := writePostingsFile(t, tc.ix)
			want := search.MatchAnd(tc.ix, tc.text)
			terms := search.QueryTerms(tc.ix, tc.text)
			for name, and := range map[string]func(...string) ([]int, error){"streaming": pf.And, "loaded": pf.AndLoaded} {
				got, err := and(terms...)
				if err != nil {
					t.Fatal(err)
				}
				if !slices.Equal(got, want) {
					t.Fatalf("%s: got %d docs %v, want %d docs %v", name, len(got), head(got), len(want), head(want))
				}
			}
		})
	}
}

func head(ids []int) []int { return ids[:min(len(ids), 5)] }

// BenchmarkPostingsFile は合成データで、メモリ上の postings とファイルの大きさ、AND の時間をメモリ上 / ファイルを順に読む / ファイルを全部読んでから計算の 3 つで比べる。
// ファイルは直前に書いたばかりで OS のページキャッシュに載っているので、ディスクから読む時間は含まない。
//
//	go test -run '^$' -bench PostingsFile -benchmem ./internal/search/
func BenchmarkPostingsFile(b *testing.B) {
	queries := map[string]string{
		"rare":   benchQuery,
		"common": "w00001 w00002 w00003",
		"mixed":  "w00001 w00300",
	}
	for _, n := range []int{10_000, 100_000} {
		ix := search.New(randomDocs(n))
		pf := writePostingsFile(b, ix)
		structs, positions := search.PostingsBytes(ix)
		stats := ix.Stats()
		b.Run(fmt.Sprintf("N=%d/size", n), func(b *testing.B) {
			for b.Loop() {
				_ = pf.Size()
			}
			b.ReportMetric(float64(structs), "mem-struct-B")
			b.ReportMetric(float64(positions), "mem-pos-B")
			b.ReportMetric(float64(stats.Postings*8), "fixed-doc-tf-B")
			b.ReportMetric(float64(pf.Size()), "file-B")
			b.ReportMetric(float64(stats.Postings), "postings")
		})
		for qn, text := range queries {
			terms := search.QueryTerms(ix, text)
			modes := []struct {
				name string
				and  func() ([]int, error)
			}{
				{name: "memory", and: func() ([]int, error) { return search.MatchAnd(ix, text), nil }},
				{name: "file-stream", and: func() ([]int, error) { return pf.And(terms...) }},
				{name: "file-loaded", and: func() ([]int, error) { return pf.AndLoaded(terms...) }},
			}
			for _, m := range modes {
				b.Run(fmt.Sprintf("N=%d/%s/%s", n, qn, m.name), func(b *testing.B) {
					b.ReportAllocs()
					var hits int
					for b.Loop() {
						ids, err := m.and()
						if err != nil {
							b.Fatal(err)
						}
						hits = len(ids)
					}
					b.ReportMetric(float64(hits), "hits/op")
				})
			}
		}
	}
}
