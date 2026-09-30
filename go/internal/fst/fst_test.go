package fst_test

import (
	"errors"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"

	"github.com/hiro8ma/agent/go/internal/fst"
)

var curry = []string{"カツ", "カツカレー", "カツ丼", "カレー", "カレーパン", "カレーライス", "キーマカレー"}

func build(t testing.TB, keys []string) *fst.FST {
	t.Helper()
	outs := make([]uint64, len(keys))
	for i := range keys {
		outs[i] = uint64(i * 7)
	}
	f, err := fst.Build(keys, outs)
	if err != nil {
		t.Fatal(err)
	}
	return f
}

// morphemeWords は形態素をつないだ語を作る。先頭と末尾の両方に共通部分が出るので、FST の共有が効く語彙になる。
func morphemeWords(n int, seed uint64) []string {
	parts := []string{
		"カレー", "カツ", "ライス", "パン", "キーマ", "チキン", "ビーフ", "ポーク", "野菜", "チーズ", "スープ", "ドリア",
		"search", "index", "query", "token", "match", "score", "rank", "term", "doc", "field", "fuzzy", "prefix",
	}
	r := rand.New(rand.NewPCG(seed, 1))
	seen := make(map[string]bool)
	for len(seen) < n {
		var b strings.Builder
		for range 1 + r.IntN(4) {
			b.WriteString(parts[r.IntN(len(parts))])
		}
		seen[b.String()] = true
	}
	words := make([]string, 0, n)
	for w := range seen {
		words = append(words, w)
	}
	sortRunes(words)
	return words
}

func sortRunes(words []string) {
	slices.SortFunc(words, func(a, b string) int { return slices.Compare([]rune(a), []rune(b)) })
}

// trieStates はトライ木の状態の数（空の語を含む異なる接頭辞の数）を数える。
func trieStates(words []string) int {
	prefixes := map[string]bool{"": true}
	for _, w := range words {
		r := []rune(w)
		for i := 1; i <= len(r); i++ {
			prefixes[string(r[:i])] = true
		}
	}
	return len(prefixes)
}

func TestGetReturnsAddedValue(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		keys []string
	}{
		"カレーの例":     {keys: slices.Clone(curry)},
		"形態素をつないだ語": {keys: morphemeWords(3000, 1)},
		"空の語を含む":    {keys: []string{"", "a", "ab", "b"}},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			sortRunes(tc.keys)
			f := build(t, tc.keys)
			if f.Len() != len(tc.keys) {
				t.Fatalf("Len = %d, want %d", f.Len(), len(tc.keys))
			}
			for i, k := range tc.keys {
				got, ok := f.Get(k)
				if !ok || got != uint64(i*7) {
					t.Errorf("Get(%q) = %d, %v, want %d", k, got, ok, i*7)
				}
			}
			for _, miss := range []string{"カ", "カレ", "ハヤシライス", "zzz"} {
				if _, ok := f.Get(miss); ok {
					t.Errorf("Get(%q) found a key that was not added", miss)
				}
			}
		})
	}
}

func TestGetWithUnorderedValues(t *testing.T) {
	t.Parallel()
	// 値が語の順に増えないと、共通の接頭辞の辺から余りを先へ押し出す処理が要る。
	keys := morphemeWords(3000, 8)
	r := rand.New(rand.NewPCG(9, 9))
	outs := make([]uint64, len(keys))
	for i := range outs {
		outs[i] = r.Uint64N(1 << 40)
	}
	f, err := fst.Build(keys, outs)
	if err != nil {
		t.Fatal(err)
	}
	for i, k := range keys {
		if got, ok := f.Get(k); !ok || got != outs[i] {
			t.Errorf("Get(%q) = %d, %v, want %d", k, got, ok, outs[i])
		}
	}
}

func TestBuildRejectsUnsortedKeys(t *testing.T) {
	t.Parallel()
	b := fst.NewBuilder()
	if err := b.Add("カレー", 1); err != nil {
		t.Fatal(err)
	}
	for _, k := range []string{"カツ", "カレー"} {
		if err := b.Add(k, 2); !errors.Is(err, fst.ErrOrder) {
			t.Errorf("Add(%q) = %v, want ErrOrder", k, err)
		}
	}
}

func TestSuffixSharingShrinksStates(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		keys []string
	}{
		"カツカレーとキーマカレー":   {keys: []string{"カツカレー", "キーマカレー"}},
		"カレーの例":          {keys: slices.Clone(curry)},
		"形態素をつないだ語 1 万":  {keys: morphemeWords(10000, 2)},
		"形態素をつないだ語 10 万": {keys: morphemeWords(100000, 2)},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			sortRunes(tc.keys)
			f := build(t, tc.keys)
			trie := trieStates(tc.keys)
			if f.States() >= trie {
				t.Errorf("FST states = %d, want fewer than trie states %d", f.States(), trie)
			}
			t.Logf("語 %d、トライ木の状態 %d、FST の状態 %d、辺 %d", len(tc.keys), trie, f.States(), f.Transitions())
		})
	}
}

func TestSharedSuffixKeepsDistinctValues(t *testing.T) {
	t.Parallel()
	f := build(t, []string{"カツカレー", "キーマカレー"})
	// カツとキーマの後ろ、続く「カレー」を 1 本の経路にまとめ、出力の和で値を分ける。トライ木なら状態は 1+5+6=12。
	if f.States() != 8 {
		t.Errorf("States = %d, want 8", f.States())
	}
	for k, want := range map[string]uint64{"カツカレー": 0, "キーマカレー": 7} {
		if got, ok := f.Get(k); !ok || got != want {
			t.Errorf("Get(%q) = %d, %v, want %d", k, got, ok, want)
		}
	}
}

func TestPrefix(t *testing.T) {
	t.Parallel()
	keys := slices.Clone(curry)
	sortRunes(keys)
	f := build(t, keys)
	testCases := map[string]struct {
		prefix string
		limit  int
		want   []string
	}{
		"カで始まる語":   {prefix: "カ", want: []string{"カツ", "カツカレー", "カツ丼", "カレー", "カレーパン", "カレーライス"}},
		"カレーで始まる語": {prefix: "カレー", want: []string{"カレー", "カレーパン", "カレーライス"}},
		"上限 2 件":   {prefix: "カ", limit: 2, want: []string{"カツ", "カツカレー"}},
		"一致なし":     {prefix: "ハヤシ", want: nil},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			var got []string
			for _, e := range f.Prefix(tc.prefix, tc.limit) {
				got = append(got, e.Key)
				if want, _ := f.Get(e.Key); e.Out != want {
					t.Errorf("Prefix の %q の値 = %d, want %d", e.Key, e.Out, want)
				}
			}
			if !slices.Equal(got, tc.want) {
				t.Errorf("Prefix(%q) = %v, want %v", tc.prefix, got, tc.want)
			}
		})
	}
}
