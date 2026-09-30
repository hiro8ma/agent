package search

import (
	"slices"
	"strings"
	"unicode"
	"unicode/utf8"
)

// DefaultCompletions は Autocomplete で返す候補の数の既定。
const DefaultCompletions = 10

// Autocomplete の K は返す候補の数で、0 なら DefaultCompletions。Counts は候補を並べる人気の数で、nil なら索引語の文書頻度を使う。
// Counts に無い語は 0 とし、同じ数なら語の文字の順に並べる。
// NoCache なら構築のときに節点ごとの上位 K 件を持たず、入力のたびに接頭辞の下の語をすべて数えて上位 K 件を選ぶ。
type Autocomplete struct {
	K       int
	Counts  map[string]int
	NoCache bool
}

// Completion の Text は入力の前の語に候補をつないだもの、Term は最後の語の候補の索引語、Count はその人気の数。
type Completion struct {
	Text  string
	Term  string
	Count int
}

// Completer は入力途中の文字列の最後の語を接頭辞として、索引語の候補を人気の順に返す。前の語は補完せず、そのまま残す。
// 候補は索引語なので、日本語の bigram で索引した語彙では 2 文字の断片が返る。語の単位で補完するには語で分ける Analyzer で索引する。
type Completer struct {
	ix     *Index
	k      int
	counts []int
	// top は節点ごとの上位の語の ID を行きがけ順に詰めた列で、節点 n の分は top[off[n]:off[n+1]]。NoCache なら nil。
	off []int32
	top []int32
}

// NewCompleter は索引の語彙の trie に、節点ごとの上位 K 件を持たせる。
func NewCompleter(ix *Index, opt Autocomplete) *Completer {
	c := &Completer{ix: ix, k: opt.K, counts: make([]int, len(ix.words))}
	if c.k <= 0 {
		c.k = DefaultCompletions
	}
	for id, w := range ix.words {
		if opt.Counts != nil {
			c.counts[id] = opt.Counts[w]
		} else {
			c.counts[id] = len(ix.postings[id])
		}
	}
	if !opt.NoCache {
		c.buildCache()
	}
	return c
}

func (c *Completer) better(a, b int32) int {
	if c.counts[a] != c.counts[b] {
		return c.counts[b] - c.counts[a]
	}
	return strings.Compare(c.ix.words[a], c.ix.words[b])
}

// buildCache は子の番号が親より大きい行きがけ順を逆にたどり、子の上位 K 件と自分の語を合わせて親の上位 K 件にする。
func (c *Completer) buildCache() {
	vt := c.ix.trie
	c.off = make([]int32, len(vt.nodes)+1)
	for n, node := range vt.nodes {
		c.off[n+1] = c.off[n] + min(int32(c.k), node.hi-node.lo)
	}
	c.top = make([]int32, c.off[len(vt.nodes)])
	var buf []int32
	for n, node := range slices.Backward(vt.nodes) {
		dst := c.top[c.off[n]:c.off[n+1]]
		if node.hi-node.lo == 1 {
			dst[0] = vt.sorted[node.lo]
			continue
		}
		buf = buf[:0]
		if node.term >= 0 {
			buf = append(buf, node.term)
		}
		for k := node.first; k < node.first+node.n; k++ {
			ch := vt.edges[k].child
			buf = append(buf, c.top[c.off[ch]:c.off[ch+1]]...)
		}
		slices.SortFunc(buf, c.better)
		copy(dst, buf)
	}
}

// Complete は typed の最後の語を接頭辞として候補を返す。typed が空白で終われば最後の語も入力済みとして扱い、候補を返さない。
func (c *Completer) Complete(typed string) []Completion {
	a := c.ix.analyzer
	text := a.normalizeText(typed)
	if last, _ := utf8.DecodeLastRuneInString(text); text == "" || unicode.IsSpace(last) {
		return nil
	}
	words := a.tokenize(text)
	if len(words) == 0 {
		return nil
	}
	var exact []string
	for _, t := range a.analyzeNormalized(strings.Join(words[:len(words)-1], " ")) {
		exact = append(exact, t.term)
	}
	head := strings.Join(exact, " ")
	if head != "" {
		head += " "
	}
	ids := c.candidates(words[len(words)-1])
	out := make([]Completion, len(ids))
	for i, id := range ids {
		w := c.ix.words[id]
		out[i] = Completion{Text: head + w, Term: w, Count: c.counts[id]}
	}
	return out
}

// candidates は prefix で始まる語の上位 K 件の ID を返す。
func (c *Completer) candidates(prefix string) []int32 {
	vt := c.ix.trie
	n, ok := vt.walk(prefix)
	if !ok {
		return nil
	}
	if c.top != nil {
		return c.top[c.off[n]:c.off[n+1]]
	}
	node := vt.nodes[n]
	top := make([]int32, 0, c.k+1)
	for _, id := range vt.sorted[node.lo:node.hi] {
		if len(top) == c.k && c.better(id, top[len(top)-1]) >= 0 {
			continue
		}
		i, _ := slices.BinarySearchFunc(top, id, c.better)
		top = slices.Insert(top, i, id)
		if len(top) > c.k {
			top = top[:c.k]
		}
	}
	return top
}
