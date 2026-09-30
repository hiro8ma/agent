package search

import (
	"fmt"
	"math/rand/v2"
	"runtime"
	"slices"
	"strings"
	"testing"
)

var vocabBenchSizes = []int{1_000, 10_000, 100_000, 1_000_000}

var benchWords = map[int][]string{}

func wordsOf(n int) []string {
	if w, ok := benchWords[n]; ok {
		return w
	}
	benchWords[n] = randomWords(n)
	return benchWords[n]
}

// heapOf は build が返したものを保持したまま GC したときに増えたヒープの大きさを測る。
func heapOf(build func() any) uint64 {
	var before, after runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&before)
	v := build()
	runtime.GC()
	runtime.ReadMemStats(&after)
	runtime.KeepAlive(v)
	return after.HeapAlloc - before.HeapAlloc
}

func vocabMap(words []string) map[string]int32 {
	m := make(map[string]int32)
	for id, w := range words {
		m[w] = int32(id)
	}
	return m
}

// BenchmarkVocabBuild は語彙から完全一致のマップ、並べた ID の列、trie（並べた ID の列を含む）を作る時間と、保持したヒープの大きさを出す。語の文字列の分は含めない。
//
//	go test -run '^$' -bench VocabBuild ./internal/search/
func BenchmarkVocabBuild(b *testing.B) {
	for _, n := range vocabBenchSizes {
		words := wordsOf(n)
		builds := []struct {
			name string
			f    func() any
		}{
			{name: "map", f: func() any { return vocabMap(words) }},
			{name: "sorted", f: func() any {
				ids := make([]int32, len(words))
				for i := range ids {
					ids[i] = int32(i)
				}
				slices.SortFunc(ids, func(a, b int32) int { return strings.Compare(words[a], words[b]) })
				return ids
			}},
			{name: "trie", f: func() any { return newVocabTrie(words) }},
		}
		for _, bd := range builds {
			b.Run(fmt.Sprintf("vocab=%d/%s", n, bd.name), func(b *testing.B) {
				for b.Loop() {
					bd.f()
				}
				b.ReportMetric(float64(heapOf(bd.f)), "heap-B")
				if bd.name == "trie" {
					vt := newVocabTrie(words)
					b.ReportMetric(float64(len(vt.nodes)), "nodes")
				}
			})
		}
	}
}

// BenchmarkVocabLookup は語彙にある語 1 語をマップ、trie、並べた ID の列の二分探索で引く。
func BenchmarkVocabLookup(b *testing.B) {
	for _, n := range vocabBenchSizes {
		words := wordsOf(n)
		m, vt := vocabMap(words), newVocabTrie(words)
		r := rand.New(rand.NewPCG(1, 1))
		probe := make([]string, 1024)
		for i := range probe {
			probe[i] = words[r.IntN(n)]
		}
		methods := []struct {
			name string
			f    func(string) int32
		}{
			{name: "map", f: func(s string) int32 { return m[s] }},
			{name: "trie", f: func(s string) int32 { id, _ := vt.lookup(s); return id }},
			{name: "sorted", f: func(s string) int32 { id, _ := vt.sortedLookup(words, s); return id }},
		}
		for _, mt := range methods {
			b.Run(fmt.Sprintf("vocab=%d/%s", n, mt.name), func(b *testing.B) {
				var sink int32
				i := 0
				for b.Loop() {
					sink += mt.f(probe[i&1023])
					i++
				}
				_ = sink
			})
		}
	}
}

// BenchmarkVocabPrefix は接頭辞で始まる語の ID を集める時間を比べる。linear は今までの alternatives と同じく全語に typoDistance をかける。
// sorted と trie は範囲を引いて ID を写すだけで、文字の順に並ぶ。trie-by-id は alternatives が使う、ID の昇順に並べ直した版。
func BenchmarkVocabPrefix(b *testing.B) {
	for _, n := range vocabBenchSizes {
		words := wordsOf(n)
		vt := newVocabTrie(words)
		for _, plen := range []int{1, 2, 4} {
			prefix := words[n/2][:plen]
			q := []rune(prefix)
			methods := []struct {
				name string
				f    func() []int32
			}{
				{name: "linear", f: func() []int32 {
					var out []int32
					for id, w := range words {
						if _, ok := typoDistance(q, w, 0, true); ok {
							out = append(out, int32(id))
						}
					}
					return out
				}},
				{name: "sorted", f: func() []int32 {
					lo, hi := vt.sortedPrefixRange(words, prefix)
					return slices.Clone(vt.sorted[lo:hi])
				}},
				{name: "trie", f: func() []int32 {
					lo, hi := vt.prefixRange(prefix)
					return slices.Clone(vt.sorted[lo:hi])
				}},
				{name: "trie-by-id", f: func() []int32 { return vt.prefixTerms(words, prefix) }},
			}
			for _, mt := range methods {
				b.Run(fmt.Sprintf("vocab=%d/len=%d/%s", n, plen, mt.name), func(b *testing.B) {
					var got []int32
					for b.Loop() {
						got = mt.f()
					}
					b.ReportMetric(float64(len(got)), "terms")
				})
			}
		}
	}
}

// BenchmarkVocabTypo は打ち間違いを許して一致する語を、全語に typoDistance をかける版と trie を降りる版で探す。
// クエリの語は語彙にある語の真ん中の 1 文字を q に置き換えたもの。
func BenchmarkVocabTypo(b *testing.B) {
	for _, n := range vocabBenchSizes {
		words := wordsOf(n)
		vt := newVocabTrie(words)
		pick := func(l int) string {
			for _, w := range words {
				if len(w) == l {
					return w[:l/2] + "q" + w[l/2+1:]
				}
			}
			return ""
		}
		cases := []struct {
			name   string
			word   string
			limit  int
			prefix bool
		}{
			{name: "len=5/typo=1", word: pick(5), limit: 1},
			{name: "len=9/typo=2", word: pick(9), limit: 2},
			{name: "len=6/typo=1+prefix", word: pick(8)[:6], limit: 1, prefix: true},
		}
		for _, c := range cases {
			q := []rune(c.word)
			methods := []struct {
				name string
				f    func() []typoMatch
			}{
				{name: "linear", f: func() []typoMatch { return linearTypoTerms(words, q, c.limit, c.prefix) }},
				{name: "trie", f: func() []typoMatch { return vt.typoTerms(words, q, c.limit, c.prefix) }},
			}
			for _, mt := range methods {
				b.Run(fmt.Sprintf("vocab=%d/%s/%s", n, c.name, mt.name), func(b *testing.B) {
					var got []typoMatch
					for b.Loop() {
						got = mt.f()
					}
					b.ReportMetric(float64(len(got)), "terms")
				})
			}
		}
	}
}

// BenchmarkCompleter は接頭辞の上位 10 件を、構築のときに節点ごとに持った版と、入力のたびに接頭辞の下の語をすべて数える版で比べる。
// 人気の数は語ごとに Zipf 分布で与える。build は Completer を作る時間で、heap-B はそれが保持するヒープ（trie は含めない）。
func BenchmarkCompleter(b *testing.B) {
	for _, n := range vocabBenchSizes {
		words := wordsOf(n)
		r := rand.New(rand.NewPCG(2, uint64(n)))
		zipf := rand.NewZipf(r, 1.1, 1, 1_000_000)
		counts := make(map[string]int, n)
		for _, w := range words {
			counts[w] = int(zipf.Uint64())
		}
		ix := &Index{words: words, trie: newVocabTrie(words), postings: make([][]posting, n), analyzer: NewAnalyzer()}
		for _, noCache := range []bool{false, true} {
			name := "cache"
			if noCache {
				name = "on-demand"
			}
			opt := Autocomplete{K: 10, Counts: counts, NoCache: noCache}
			b.Run(fmt.Sprintf("vocab=%d/build/%s", n, name), func(b *testing.B) {
				for b.Loop() {
					NewCompleter(ix, opt)
				}
				b.ReportMetric(float64(heapOf(func() any { return NewCompleter(ix, opt) })), "heap-B")
			})
			c := NewCompleter(ix, opt)
			for _, plen := range []int{1, 2, 4} {
				prefix := words[n/2][:plen]
				lo, hi := ix.trie.prefixRange(prefix)
				b.Run(fmt.Sprintf("vocab=%d/len=%d/%s", n, plen, name), func(b *testing.B) {
					for b.Loop() {
						c.candidates(prefix)
					}
					b.ReportMetric(float64(hi-lo), "subtree")
				})
			}
		}
	}
}
