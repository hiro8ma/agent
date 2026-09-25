package search

import (
	"math/rand/v2"
	"strings"
	"testing"
)

func TestTypoDistance(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		query, term string
		limit       int
		prefix      bool
		want        int
		wantOK      bool
	}{
		"1 文字の置換は 1":             {query: "sewrch", term: "search", limit: 1, want: 1, wantOK: true},
		"1 文字の挿入は 1":             {query: "seearch", term: "search", limit: 1, want: 1, wantOK: true},
		"1 文字の削除は 1":             {query: "serch", term: "search", limit: 1, want: 1, wantOK: true},
		"隣り合う 2 文字の入れ替えは 1":      {query: "serach", term: "search", limit: 1, want: 1, wantOK: true},
		"最初の文字の置換は 2 と数える":       {query: "xearch", term: "search", limit: 1, want: 2, wantOK: false},
		"最初の文字の置換は 2 つまでなら許す":    {query: "xetrieval", term: "retrieval", limit: 2, want: 2, wantOK: true},
		"長さが上限より離れた語は比べない":       {query: "search", term: "searching", limit: 1, wantOK: false},
		"接頭辞なら続きの文字は数えない":        {query: "searc", term: "searching", limit: 1, prefix: true, want: 0, wantOK: true},
		"接頭辞でも打ち間違いを数える":         {query: "serac", term: "searching", limit: 1, prefix: true, want: 1, wantOK: true},
		"上限が 0 なら接頭辞が一致する語だけを拾う": {query: "sea", term: "seashell", limit: 0, prefix: true, want: 0, wantOK: true},
		"上限が 0 なら接頭辞の打ち間違いは拾わない": {query: "sae", term: "seashell", limit: 0, prefix: true, wantOK: false},
		"上限が 0 で接頭辞でなければ拾わない":    {query: "sea", term: "seashell", limit: 0, wantOK: false},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			got, ok := typoDistance([]rune(tc.query), tc.term, tc.limit, tc.prefix)
			if ok != tc.wantOK || (ok && got != tc.want) {
				t.Fatalf("typoDistance(%q, %q) = %d, %v, want %d, %v", tc.query, tc.term, got, ok, tc.want, tc.wantOK)
			}
		})
	}
}

func TestMaxTypos(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		runes int
		want  int
	}{
		"4 文字は許さない":    {runes: 4, want: 0},
		"5 文字から 1 つ許す": {runes: 5, want: 1},
		"8 文字も 1 つ":    {runes: 8, want: 1},
		"9 文字から 2 つ許す": {runes: 9, want: 2},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			if got := maxTypos(tc.runes); got != tc.want {
				t.Fatalf("maxTypos(%d) = %d, want %d", tc.runes, got, tc.want)
			}
		})
	}
}

// randomWordIndex は a から z の 3 から 12 文字の重複しない語を n 語作り、10 語ずつ文書にして索引する。
func randomWordIndex(n int) *Index {
	r := rand.New(rand.NewPCG(5, uint64(n)))
	seen := make(map[string]struct{}, n)
	ds := make([]Doc, 0, n/10)
	var words []string
	for len(seen) < n {
		b := make([]byte, 3+r.IntN(10))
		for i := range b {
			b[i] = byte('a' + r.IntN(26))
		}
		if _, ok := seen[string(b)]; ok {
			continue
		}
		seen[string(b)] = struct{}{}
		words = append(words, string(b))
		if len(words) == 10 {
			ds = append(ds, Doc{Content: strings.Join(words, " ")})
			words = words[:0]
		}
	}
	return New(ds)
}

// BenchmarkTypoExpansion は語彙目録を全件なめて、1 語に一致させる索引語を探す時間と見つかった数を出す。
// クエリの語は語彙にある語の 1 文字を置き換えたもの。
//
//	go test -run '^$' -bench TypoExpansion ./internal/search/
func BenchmarkTypoExpansion(b *testing.B) {
	ix := randomWordIndex(100_000)
	pick := func(n int) string {
		for _, w := range ix.words {
			if len(w) == n {
				return w[:n/2] + "q" + w[n/2+1:]
			}
		}
		return ""
	}
	cases := []struct {
		name string
		word string
		q    Query
		last bool
	}{
		{name: "len=5/typo", word: pick(5), q: Query{Typo: true}},
		{name: "len=9/typo", word: pick(9), q: Query{Typo: true}},
		{name: "len=3/prefix", word: ix.words[0][:3], q: Query{Prefix: true}, last: true},
		{name: "len=6/typo+prefix", word: pick(8)[:6], q: Query{Typo: true, Prefix: true}, last: true},
	}
	for _, c := range cases {
		b.Run(c.name, func(b *testing.B) {
			var alts []alt
			for b.Loop() {
				n := 0
				alts = ix.alternatives(c.word, c.q, c.last, func(string) int { n++; return n })
			}
			b.ReportMetric(float64(len(alts)-1), "alts")
			b.ReportMetric(float64(len(ix.words)), "vocab")
		})
	}
}

// TestJapaneseBigramTypoExpansion は日本語の bigram に打ち間違いを 1 つ許した場合と、1 文字のクエリを接頭辞で広げた場合に、一致させる索引語がいくつ増えるかを出す。
// 既定では alternatives が日本語を広げないので、typoDistance を直接呼んで数える。
func TestJapaneseBigramTypoExpansion(t *testing.T) {
	t.Parallel()
	r := rand.New(rand.NewPCG(7, 10_000))
	particles := []string{"の", "は", "が", "を", "に", "で", "と", "も", "には", "では", "への"}
	ds := make([]Doc, 10_000)
	for i := range ds {
		var sb strings.Builder
		for range 10 + r.IntN(31) {
			k := r.IntN(20_000)
			sb.WriteString(string([]rune{rune(0x4E00 + k/200), rune(0x4F00 + k%200)}))
			sb.WriteString(particles[r.IntN(len(particles))])
		}
		ds[i] = Doc{Content: sb.String()}
	}
	ix := New(ds)
	bigram := string([]rune{rune(0x4E00), rune(0x4F00 + 50)})
	testCases := map[string]struct {
		query  string
		limit  int
		prefix bool
	}{
		"bigram に打ち間違いを 1 つ許す": {query: bigram, limit: 1},
		"漢字 1 文字を接頭辞で広げる":      {query: string(rune(0x4E00)), prefix: true},
		"助詞 1 文字を接頭辞で広げる":      {query: "に", prefix: true},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			n := 0
			for _, w := range ix.words {
				if _, ok := typoDistance([]rune(tc.query), w, tc.limit, tc.prefix); ok && w != tc.query {
					n++
				}
			}
			t.Logf("%s: %d of %d terms", tc.query, n, len(ix.words))
			if n == 0 {
				t.Fatal("no expansion")
			}
		})
	}
}
