package search

import (
	"fmt"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"
	"unicode/utf8"
)

// randomWords は a から z の 3 から 12 文字の重複しない語を n 語、固定の種で作る。
func randomWords(n int) []string {
	r := rand.New(rand.NewPCG(11, uint64(n)))
	seen := make(map[string]struct{}, n)
	words := make([]string, 0, n)
	for len(words) < n {
		b := make([]byte, 3+r.IntN(10))
		for i := range b {
			b[i] = byte('a' + r.IntN(26))
		}
		if _, ok := seen[string(b)]; ok {
			continue
		}
		seen[string(b)] = struct{}{}
		words = append(words, string(b))
	}
	return words
}

// mixedWords は英字の語に、日本語の語と、UTF-8 として正しくない語を混ぜる。
func mixedWords() []string {
	return append(randomWords(5_000),
		"カツ", "カツカレー", "カツ丼", "カレー", "カレーパン", "カレーライス", "カ", "スイカ",
		"se\xffa", "sea\xff", "\xffsea", "search", "searching", "seashell", "sea", "serach")
}

func linearPrefixTerms(words []string, prefix string) []int32 {
	var out []int32
	for id, w := range words {
		if strings.HasPrefix(w, prefix) {
			out = append(out, int32(id))
		}
	}
	return out
}

func linearTypoTerms(words []string, q []rune, limit int, prefix bool) []typoMatch {
	var out []typoMatch
	for id, w := range words {
		if d, ok := typoDistance(q, w, limit, prefix); ok {
			out = append(out, typoMatch{term: int32(id), typos: d})
		}
	}
	return out
}

func TestVocabTrieLookup(t *testing.T) {
	t.Parallel()
	words := mixedWords()
	vt := newVocabTrie(words)
	for id, w := range words {
		want, valid := int32(id), utf8.ValidString(w)
		got, ok := vt.lookup(w)
		sgot, sok := vt.sortedLookup(words, w)
		if valid && (!ok || got != want || !sok || sgot != want) {
			t.Fatalf("%q: trie %d %v, sorted %d %v, want %d", w, got, ok, sgot, sok, want)
		}
	}
	for _, w := range []string{"", "zzzzzzzzzzzzzz", "カツカ", "カレーラ", "seashel"} {
		if id, ok := vt.lookup(w); ok {
			t.Fatalf("%q: found %d", w, id)
		}
	}
}

func TestVocabTriePrefixMatchesLinearScan(t *testing.T) {
	t.Parallel()
	words := mixedWords()
	vt := newVocabTrie(words)
	var prefixes []string
	for _, w := range words[:300] {
		r := []rune(w)
		for n := 1; n <= min(len(r), 4); n++ {
			prefixes = append(prefixes, string(r[:n]))
		}
	}
	prefixes = append(prefixes, "", "カ", "カレ", "カレー", "カツ", "se", "sea", "qqqqq", "カツカレーパン")
	for _, p := range prefixes {
		want := linearPrefixTerms(words, p)
		if got := vt.prefixTerms(words, p); !slices.Equal(got, want) {
			t.Fatalf("trie %q: %d terms, want %d", p, len(got), len(want))
		}
		lo, hi := vt.prefixRange(p)
		slo, shi := vt.sortedPrefixRange(words, p)
		if hi-lo != shi-slo || (hi > lo && lo != slo) {
			t.Fatalf("%q: trie [%d, %d), sorted [%d, %d)", p, lo, hi, slo, shi)
		}
	}
}

func TestVocabTrieTypoMatchesLinearScan(t *testing.T) {
	t.Parallel()
	words := mixedWords()
	vt := newVocabTrie(words)
	r := rand.New(rand.NewPCG(3, 3))
	var queries []string
	for _, w := range words[:200] {
		q := []byte(w)
		switch i := r.IntN(len(q)); r.IntN(4) {
		case 0:
			q[i] = byte('a' + r.IntN(26))
		case 1:
			q = slices.Delete(q, i, i+1)
		case 2:
			q = slices.Insert(q, i, byte('a'+r.IntN(26)))
		default:
			if i+1 < len(q) {
				q[i], q[i+1] = q[i+1], q[i]
			}
		}
		queries = append(queries, string(q), w)
	}
	queries = append(queries, "serac", "seashel", "カツカレー", "カレーライ")
	for _, q := range queries {
		qr := []rune(q)
		if len(qr) == 0 {
			continue
		}
		for _, limit := range []int{1, 2} {
			for _, prefix := range []bool{false, true} {
				want := linearTypoTerms(words, qr, limit, prefix)
				if got := vt.typoTerms(words, qr, limit, prefix); !slices.Equal(got, want) {
					t.Fatalf("%q limit %d prefix %v: trie %v, want %v", q, limit, prefix, got, want)
				}
			}
		}
	}
}

// TestJapanesePrefixOnBigrams は bigram で索引した日本語に接頭辞の一致を使うと何が一致するかを確かめる。alternatives は日本語を広げないので、trie を直接引く。
// 2 文字以上の入力は bigram の完全一致で語の途中も拾うので、接頭辞にしても増える語は無い。
// 1 文字の入力を接頭辞で広げると、その字の後ろに字が続く bigram に一致する。語の先頭に限らず、語の最後にあるその字は拾わない。
func TestJapanesePrefixOnBigrams(t *testing.T) {
	t.Parallel()
	ix := New([]Doc{
		{ID: "katsu", Content: "カツ丼"},
		{ID: "katsucurry", Content: "カツカレー"},
		{ID: "bread", Content: "カレーパン"},
		{ID: "watermelon", Content: "スイカ"},
		{ID: "ka", Content: "カ"},
	})
	ids := func(q Query) []string {
		var out []string
		for _, h := range ix.RankQuery(q, -1).Hits {
			out = append(out, h.Doc.ID)
		}
		slices.Sort(out)
		return out
	}
	expanded := func(prefix string) []string {
		var out []string
		for _, id := range ix.trie.prefixTerms(ix.words, prefix) {
			out = append(out, ix.words[id])
		}
		return out
	}
	testCases := map[string]struct {
		prefix    string
		wantTerms []string
		wantDocs  []string
	}{
		"2 文字の入力は bigram 1 語だけで語の途中も一致する": {
			prefix: "カレ", wantTerms: []string{"カレ"}, wantDocs: []string{"bread", "katsucurry"},
		},
		"1 文字の入力を広げると語の最後のカを含むスイカは一致しない": {
			prefix: "カ", wantTerms: []string{"カ", "カツ", "カレ"}, wantDocs: []string{"bread", "ka", "katsu", "katsucurry"},
		},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			terms := expanded(tc.prefix)
			slices.Sort(terms)
			var docs []string
			for _, w := range terms {
				docs = append(docs, ids(Query{Text: w})...)
			}
			slices.Sort(docs)
			docs = slices.Compact(docs)
			if !slices.Equal(terms, tc.wantTerms) || !slices.Equal(docs, tc.wantDocs) {
				t.Fatalf("terms %v docs %v, want %v %v", terms, docs, tc.wantTerms, tc.wantDocs)
			}
			if got := ids(Query{Text: tc.prefix, Prefix: true}); !slices.Equal(got, ids(Query{Text: tc.prefix})) {
				t.Fatalf("Prefix changed Japanese results: %v", got)
			}
		})
	}
}

func TestCompleterCacheMatchesOnDemand(t *testing.T) {
	t.Parallel()
	words := mixedWords()
	r := rand.New(rand.NewPCG(9, 9))
	counts := make(map[string]int, len(words))
	for _, w := range words {
		counts[w] = r.IntN(20)
	}
	ix := &Index{words: words, trie: newVocabTrie(words), postings: make([][]posting, len(words)), analyzer: NewAnalyzer()}
	for _, k := range []int{1, 3, 10} {
		t.Run(fmt.Sprintf("K=%d", k), func(t *testing.T) {
			t.Parallel()
			cached := NewCompleter(ix, Autocomplete{K: k, Counts: counts})
			onDemand := NewCompleter(ix, Autocomplete{K: k, Counts: counts, NoCache: true})
			for _, w := range words[:500] {
				for n := 1; n <= min(len(w), 3); n++ {
					p := w[:n]
					got, want := cached.candidates(p), onDemand.candidates(p)
					if !slices.Equal(got, want) {
						t.Fatalf("%q: cached %v, on demand %v", p, got, want)
					}
					if lo, hi := ix.trie.prefixRange(p); len(want) != min(k, int(hi-lo)) {
						t.Fatalf("%q: %d candidates of %d terms", p, len(want), hi-lo)
					}
				}
			}
		})
	}
}
