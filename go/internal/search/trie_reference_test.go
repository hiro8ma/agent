package search

import (
	"slices"
	"strings"
	"testing"
	"unicode/utf8"
)

// refTrie は教材どおりに節点をポインタでつないだ素朴なトライ木。vocabTrie の結果を確かめる基準にする。
// vocabTrie は同じ木を配列に並べ、節点ごとに下の語の範囲を持たせて速くしたもの。
type refTrie struct {
	children map[rune]*refTrie
	term     bool
}

func newRefTrie(words ...string) *refTrie {
	t := &refTrie{}
	for _, w := range words {
		t.insert(w)
	}
	return t
}

// insert は語を 1 文字ずつたどり、無い節点を作りながら降りて、最後の節点に語の終わりの印を付ける。
func (t *refTrie) insert(word string) {
	n := t
	for _, r := range word {
		if n.children == nil {
			n.children = make(map[rune]*refTrie)
		}
		next, ok := n.children[r]
		if !ok {
			next = &refTrie{}
			n.children[r] = next
		}
		n = next
	}
	n.term = true
}

// find は s の文字をたどった先の節点を返す。かかる手間は語彙の数ではなく s の長さで決まる。
func (t *refTrie) find(s string) *refTrie {
	n := t
	for _, r := range s {
		next, ok := n.children[r]
		if !ok {
			return nil
		}
		n = next
	}
	return n
}

func (t *refTrie) contains(word string) bool {
	n := t.find(word)
	return n != nil && n.term
}

// withPrefix は prefix の節点まで降り、その下にある語を文字の順にすべて集める。
func (t *refTrie) withPrefix(prefix string) []string {
	n := t.find(prefix)
	if n == nil {
		return nil
	}
	var out []string
	var walk func(n *refTrie, word []rune)
	walk = func(n *refTrie, word []rune) {
		if n.term {
			out = append(out, string(word))
		}
		for _, r := range sortedKeys(n.children) {
			walk(n.children[r], append(word, r))
		}
	}
	walk(n, []rune(prefix))
	return out
}

// draw は木の形を文字で描く。語の終わりの節点には ← と語を添える。
func (t *refTrie) draw() string {
	var b strings.Builder
	b.WriteString("ROOT\n")
	var walk func(n *refTrie, word []rune, indent string)
	walk = func(n *refTrie, word []rune, indent string) {
		keys := sortedKeys(n.children)
		for i, r := range keys {
			child := n.children[r]
			branch, next := "├─ ", "│  "
			if i == len(keys)-1 {
				branch, next = "└─ ", "   "
			}
			w := append(slices.Clone(word), r)
			b.WriteString(indent + branch + string(r))
			if child.term {
				b.WriteString("  ← " + string(w))
			}
			b.WriteString("\n")
			walk(child, w, indent+next)
		}
	}
	walk(t, nil, "")
	return b.String()
}

func sortedKeys(m map[rune]*refTrie) []rune {
	keys := make([]rune, 0, len(m))
	for r := range m {
		keys = append(keys, r)
	}
	slices.Sort(keys)
	return keys
}

var curryWords = []string{"カツ", "カツカレー", "カツ丼", "カレー", "カレーパン", "カレーライス"}

func TestRefTrieDrawsSharedPrefixes(t *testing.T) {
	t.Parallel()
	// 教材の図と同じ形。カツ系とカレー系は「カ」を共有し、カレーパンとカレーライスは「カレー」まで共有する。
	want := `ROOT
└─ カ
   ├─ ツ  ← カツ
   │  ├─ カ
   │  │  └─ レ
   │  │     └─ ー  ← カツカレー
   │  └─ 丼  ← カツ丼
   └─ レ
      └─ ー  ← カレー
         ├─ パ
         │  └─ ン  ← カレーパン
         └─ ラ
            └─ イ
               └─ ス  ← カレーライス
`
	if got := newRefTrie(curryWords...).draw(); got != want {
		t.Errorf("draw() =\n%s\nwant\n%s", got, want)
	}
}

func TestRefTrieLookupAndPrefix(t *testing.T) {
	t.Parallel()
	tr := newRefTrie(curryWords...)
	testCases := map[string]struct {
		prefix     string
		wantPrefix []string
		wantWord   bool
	}{
		"カで始まる語はすべて":       {prefix: "カ", wantPrefix: curryWords, wantWord: false},
		"カレで始まる語はカレー系だけ":   {prefix: "カレ", wantPrefix: []string{"カレー", "カレーパン", "カレーライス"}, wantWord: false},
		"カレーは語で、接頭辞でもある":   {prefix: "カレー", wantPrefix: []string{"カレー", "カレーパン", "カレーライス"}, wantWord: true},
		"カツ丼は語で、下に語はない":    {prefix: "カツ丼", wantPrefix: []string{"カツ丼"}, wantWord: true},
		"辞書にない語は途中で節点が切れる": {prefix: "カツサンド", wantPrefix: nil, wantWord: false},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			if got := tr.withPrefix(tc.prefix); !slices.Equal(got, tc.wantPrefix) {
				t.Errorf("withPrefix(%q) = %v, want %v", tc.prefix, got, tc.wantPrefix)
			}
			if got := tr.contains(tc.prefix); got != tc.wantWord {
				t.Errorf("contains(%q) = %v, want %v", tc.prefix, got, tc.wantWord)
			}
		})
	}
}

// TestVocabTrieMatchesRefTrie は、配列に並べて速くした vocabTrie が、素朴なトライ木と同じ答えを返すことを確かめる。
func TestVocabTrieMatchesRefTrie(t *testing.T) {
	t.Parallel()
	var words []string
	for _, w := range mixedWords() {
		// 素朴なトライ木は文字（rune）で降りるので、文字に区切れない不正な UTF-8 の語は比べない。
		if utf8.ValidString(w) {
			words = append(words, w)
		}
	}
	vt := newVocabTrie(words)
	ref := newRefTrie(words...)

	prefixes := []string{"", "カ", "カレ", "カレー", "カツ丼", "ス", "se", "sea", "sear", "zz", "カツサンド"}
	for _, w := range words[:300] {
		r := []rune(w)
		prefixes = append(prefixes, string(r[:1]), string(r[:len(r)/2]), w, w+"x")
	}
	for _, p := range prefixes {
		var got []string
		for _, id := range vt.prefixTerms(words, p) {
			got = append(got, words[id])
		}
		slices.Sort(got)
		want := ref.withPrefix(p)
		slices.Sort(want)
		if !slices.Equal(got, want) {
			t.Errorf("prefix %q: vocabTrie %d 語, refTrie %d 語", p, len(got), len(want))
		}
		_, inVocab := vt.lookup(p)
		if inVocab != ref.contains(p) {
			t.Errorf("lookup(%q) = %v, refTrie = %v", p, inVocab, ref.contains(p))
		}
	}
}
