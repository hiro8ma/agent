package fst_test

import (
	"fmt"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/hiro8ma/agent/go/internal/fst"
)

func TestDistance(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		a, b string
		want int
	}{
		"同じ語":       {a: "カレーライス", b: "カレーライス", want: 0},
		"置換 1 回":    {a: "カレーライス", b: "カレーレイス", want: 1},
		"挿入 1 回":    {a: "カレーライス", b: "カレーライース", want: 1},
		"削除 1 回":    {a: "カレーライス", b: "カレーライ", want: 1},
		"隣の入れ替えは 2": {a: "abc", b: "acb", want: 2},
		"空の語":       {a: "", b: "abc", want: 3},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			if got := fst.Distance(tc.a, tc.b); got != tc.want {
				t.Errorf("Distance(%q, %q) = %d, want %d", tc.a, tc.b, got, tc.want)
			}
		})
	}
}

// bruteForce は語彙の全語と距離を計算する。オートマトンの結果が正しいかの基準にする。
func bruteForce(words []string, q string, k int) []string {
	var out []string
	for _, w := range words {
		if fst.Distance(q, w) <= k {
			out = append(out, w)
		}
	}
	return out
}

func keysOf(ms []fst.Match) []string {
	out := make([]string, len(ms))
	for i, m := range ms {
		out[i] = m.Key
	}
	return out
}

func TestFuzzyMatchesBruteForce(t *testing.T) {
	t.Parallel()
	words := morphemeWords(5000, 3)
	f := build(t, words)
	r := rand.New(rand.NewPCG(4, 4))
	queries := []string{"abaaba", "カレーレイス", "カツカレ", "serch", "indx", "キーマカレー"}
	for range 40 {
		// 語彙の語を 1〜2 か所壊したクエリを混ぜる。
		w := []rune(words[r.IntN(len(words))])
		for range 1 + r.IntN(2) {
			if len(w) > 1 {
				i := r.IntN(len(w))
				w = slices.Delete(w, i, i+1)
			}
		}
		queries = append(queries, string(w))
	}
	for _, k := range []int{0, 1, 2} {
		for _, q := range queries {
			t.Run(fmt.Sprintf("k=%d/%s", k, q), func(t *testing.T) {
				t.Parallel()
				want := bruteForce(words, q, k)
				dfa := f.Fuzzy(fst.NewLevenshtein(q, k))
				rows := f.FuzzyRows(q, k)
				if got := keysOf(dfa); !slices.Equal(got, want) {
					t.Errorf("Fuzzy = %v, want %v", got, want)
				}
				if got := keysOf(rows); !slices.Equal(got, want) {
					t.Errorf("FuzzyRows = %v, want %v", got, want)
				}
				for _, m := range dfa {
					if d := fst.Distance(q, m.Key); m.Dist != d {
						t.Errorf("%q の距離 = %d, want %d", m.Key, m.Dist, d)
					}
					if v, _ := f.Get(m.Key); m.Out != v {
						t.Errorf("%q の値 = %d, want %d", m.Key, m.Out, v)
					}
				}
			})
		}
	}
}

func TestLevenshteinAcceptsOneSubstitution(t *testing.T) {
	t.Parallel()
	f := build(t, []string{"abaaba", "ababba", "abbbba", "bbaaba"})
	got := keysOf(f.Fuzzy(fst.NewLevenshtein("abaaba", 1)))
	want := []string{"abaaba", "ababba", "bbaaba"}
	if !slices.Equal(got, want) {
		t.Errorf("Fuzzy(abaaba, 1) = %v, want %v", got, want)
	}
}

func TestLevenshteinReusesStates(t *testing.T) {
	t.Parallel()
	words := morphemeWords(20000, 5)
	f := build(t, words)
	l := fst.NewLevenshtein("カレーライス", 1)
	f.Fuzzy(l)
	// クエリに出てこない文字はまとめて扱うので、語彙の文字の種類によらず状態の数は小さく収まる。
	if l.States() > 100 {
		t.Errorf("DFA states = %d, want at most 100", l.States())
	}
	t.Logf("語 %d を探して、作った DFA の状態は %d", len(words), l.States())
}

func BenchmarkFuzzy(b *testing.B) {
	for _, n := range []int{10_000, 100_000} {
		words := morphemeWords(n, 6)
		f := build(b, words)
		queries := []string{"カレーレイス", "serchindx", "チキンカツカレ", "fuzy"}
		for _, k := range []int{1, 2} {
			b.Run(fmt.Sprintf("n=%d/k=%d/brute", n, k), func(b *testing.B) {
				for b.Loop() {
					for _, q := range queries {
						bruteForce(words, q, k)
					}
				}
			})
			b.Run(fmt.Sprintf("n=%d/k=%d/rows", n, k), func(b *testing.B) {
				for b.Loop() {
					for _, q := range queries {
						f.FuzzyRows(q, k)
					}
				}
			})
			b.Run(fmt.Sprintf("n=%d/k=%d/dfa", n, k), func(b *testing.B) {
				for b.Loop() {
					for _, q := range queries {
						f.Fuzzy(fst.NewLevenshtein(q, k))
					}
				}
			})
		}
	}
}

func BenchmarkBuild(b *testing.B) {
	for _, n := range []int{10_000, 100_000} {
		words := morphemeWords(n, 7)
		outs := make([]uint64, n)
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			for b.Loop() {
				if _, err := fst.Build(words, outs); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
