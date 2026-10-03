package search

import (
	"testing"
	"unicode/utf8"

	"github.com/hiro8ma/agent/go/internal/fst"
)

// TestFuzzinessMatchesMaxTypos は fst.DefaultFuzziness（1〜4 文字は完全一致、5〜8 文字は 1、9 文字以上は 2）が、この索引の Meilisearch の既定（5 と 9）と同じ距離を選ぶことを確かめる。
// Elasticsearch の AUTO の既定（AUTO:3,6）は 3〜4 文字と 6〜8 文字で 1 つ多く許すので、その長さを出す。
func TestFuzzinessMatchesMaxTypos(t *testing.T) {
	t.Parallel()
	auto := fst.Fuzziness{One: 3, Two: 6}
	for n := 1; n <= 12; n++ {
		word := string(make([]rune, n))
		if got, want := fst.DefaultFuzziness.K(word), maxTypos(n); got != want {
			t.Errorf("%d 文字: DefaultFuzziness = %d, maxTypos = %d", n, got, want)
		}
		if a := auto.K(word); a != maxTypos(n) {
			t.Logf("%2d 文字: 既定 %d / AUTO:3,6 %d", n, maxTypos(n), a)
		}
	}
}

// TestFuzzinessDistanceDiffers は許す距離が同じでも、距離の数え方の違いで一致する語が変わる例を固定する。
// typoDistance は隣り合う 2 文字の入れ替えを 1 つと数え、最初の文字の違いを 1 つ多く数える。fst はレーベンシュタイン距離で数える。
func TestFuzzinessDistanceDiffers(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		query, term string
		search, fst bool
	}{
		"入れ替えは typoDistance では 1 つ":      {query: "serach", term: "search", search: true, fst: false},
		"最初の文字の置換は typoDistance では 2 つ":  {query: "xearch", term: "search", search: false, fst: true},
		"途中の置換はどちらも 1 つ":                 {query: "seerch", term: "search", search: true, fst: true},
		"4 文字の置換はどちらも許さない":               {query: "rank", term: "rink", search: false, fst: false},
		"9 文字の入れ替え 2 つは typoDistance だけ": {query: "seacrhign", term: "searching", search: true, fst: false},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			k := fst.DefaultFuzziness.K(tc.query)
			_, s := typoDistance([]rune(tc.query), tc.term, maxTypos(utf8.RuneCountInString(tc.query)), false)
			_, f := fst.NewDFA(tc.query, k).Accept(tc.term)
			if s != tc.search || f != tc.fst {
				t.Errorf("%s→%s (k=%d): typoDistance %v / fst %v, want %v / %v", tc.query, tc.term, k, s, f, tc.search, tc.fst)
			}
		})
	}
}
