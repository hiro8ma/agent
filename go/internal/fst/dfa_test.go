package fst_test

import (
	"fmt"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/hiro8ma/agent/go/internal/fst"
)

// TestDFAMatchesLazyAndBruteForce は先に全体を作った DFA が、必要な遷移だけ作る DFA と全語なめの両方と同じ語と距離を返すことを確かめる。
func TestDFAMatchesLazyAndBruteForce(t *testing.T) {
	t.Parallel()
	words := morphemeWords(5000, 3)
	f := build(t, words)
	r := rand.New(rand.NewPCG(8, 8))
	queries := []string{"abaaba", "カレーレイス", "カツカレ", "serch", "indx", "キーマカレー", "a"}
	for range 30 {
		w := []rune(words[r.IntN(len(words))])
		if len(w) > 1 {
			i := r.IntN(len(w))
			w = slices.Delete(w, i, i+1)
		}
		queries = append(queries, string(w))
	}
	for _, k := range []int{0, 1, 2} {
		for _, q := range queries {
			t.Run(fmt.Sprintf("k=%d/%s", k, q), func(t *testing.T) {
				t.Parallel()
				d := fst.NewDFA(q, k)
				eager := f.FuzzyDFA(d)
				if lazy := f.Fuzzy(fst.NewLevenshtein(q, k)); !slices.Equal(eager, lazy) {
					t.Errorf("FuzzyDFA = %v, Fuzzy = %v", keysOf(eager), keysOf(lazy))
				}
				if got, want := keysOf(eager), bruteForce(words, q, k); !slices.Equal(got, want) {
					t.Errorf("FuzzyDFA = %v, want %v", got, want)
				}
				for _, w := range words[:500] {
					dist, ok := d.Accept(w)
					want := fst.Distance(q, w)
					if ok != (want <= k) || (ok && dist != want) {
						t.Errorf("Accept(%q) = %d, %v, want distance %d", w, dist, ok, want)
					}
				}
			})
		}
	}
}

func TestDFANegativeKAcceptsNothing(t *testing.T) {
	t.Parallel()
	d := fst.NewDFA("abc", -1)
	if _, ok := d.Accept("abc"); ok || d.States() != 0 {
		t.Errorf("k=-1: Accept = %v, States = %d, want false, 0", ok, d.States())
	}
}

func TestFuzzinessK(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		f    fst.Fuzziness
		word string
		want int
	}{
		"既定で 4 文字は完全一致":            {f: fst.DefaultFuzziness, word: "abcd", want: 0},
		"既定で 5 文字は距離 1":            {f: fst.DefaultFuzziness, word: "abcde", want: 1},
		"既定で 8 文字は距離 1":            {f: fst.DefaultFuzziness, word: "abcdefgh", want: 1},
		"既定で 9 文字は距離 2":            {f: fst.DefaultFuzziness, word: "abcdefghi", want: 2},
		"文字数は rune で数える":           {f: fst.DefaultFuzziness, word: "カレーライス", want: 1},
		"閾値を 3 と 6 にすると 3 文字で距離 1": {f: fst.Fuzziness{One: 3, Two: 6}, word: "abc", want: 1},
		"閾値を 3 と 6 にすると 6 文字で距離 2": {f: fst.Fuzziness{One: 3, Two: 6}, word: "abcdef", want: 2},
		"空の語は完全一致":                 {f: fst.DefaultFuzziness, word: "", want: 0},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			if got := tc.f.K(tc.word); got != tc.want {
				t.Errorf("K(%q) = %d, want %d", tc.word, got, tc.want)
			}
		})
	}
}

// BenchmarkFuzzyEager は、必要な遷移だけを作る DFA と、先に全体を作る DFA の、作る時間と FST と合わせてたどる時間を比べる。
//
//	go test -run '^$' -bench FuzzyEager ./internal/fst/
func BenchmarkFuzzyEager(b *testing.B) {
	words := morphemeWords(100_000, 6)
	f := build(b, words)
	queries := []string{"カレーレイス", "serchindx", "チキンカツカレ", "fuzy"}
	for _, k := range []int{1, 2} {
		b.Run(fmt.Sprintf("k=%d/lazy", k), func(b *testing.B) {
			for b.Loop() {
				for _, q := range queries {
					f.Fuzzy(fst.NewLevenshtein(q, k))
				}
			}
		})
		b.Run(fmt.Sprintf("k=%d/eager", k), func(b *testing.B) {
			for b.Loop() {
				for _, q := range queries {
					f.FuzzyDFA(fst.NewDFA(q, k))
				}
			}
		})
		b.Run(fmt.Sprintf("k=%d/eager-build-only", k), func(b *testing.B) {
			for b.Loop() {
				for _, q := range queries {
					fst.NewDFA(q, k)
				}
			}
		})
	}
}
