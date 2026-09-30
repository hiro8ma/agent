package search

import (
	"slices"
	"sort"
	"strings"
	"unicode/utf8"
)

// vocabTrie は語彙を文字（rune）単位の trie にしたもの。節点を行きがけ順に並べ、節点の下の語が sorted の連続した範囲 [lo, hi) になるようにする。
// 接頭辞の一致は、接頭辞の節点まで降りて範囲を返すだけで済み、語彙の大きさによらない。
// sorted は語の ID を文字の順に並べた列で、二分探索で接頭辞の範囲を引く比較にも使う。
// UTF-8 として正しくない語は文字の並びとバイトの並びが一致しないので trie に入れず、invalid に分けて全件を比べる。
type vocabTrie struct {
	nodes   []trieNode
	edges   []trieEdge
	sorted  []int32
	invalid []int32
}

// trieNode の子は edges[first : first+n] に文字の昇順で並ぶ。term はこの節点で終わる語の ID で、無ければ -1。
type trieNode struct {
	first, n int32
	term     int32
	lo, hi   int32
}

type trieEdge struct {
	r     rune
	child int32
}

func newVocabTrie(words []string) *vocabTrie {
	vt := &vocabTrie{}
	for id, w := range words {
		if utf8.ValidString(w) {
			vt.sorted = append(vt.sorted, int32(id))
		} else {
			vt.invalid = append(vt.invalid, int32(id))
		}
	}
	// 正しい UTF-8 ではバイトの順と文字の順が一致するので、文字列の比較で並べれば trie の行きがけ順になる。
	slices.SortFunc(vt.sorted, func(a, b int32) int { return strings.Compare(words[a], words[b]) })
	vt.build(words, 0, int32(len(vt.sorted)), 0)
	return vt
}

// build は sorted[lo:hi] の語が先頭 off バイトを共有しているとして節点を作り、その番号を返す。
func (vt *vocabTrie) build(words []string, lo, hi int32, off int) int32 {
	id := int32(len(vt.nodes))
	vt.nodes = append(vt.nodes, trieNode{term: -1, lo: lo, hi: hi})
	i := lo
	if i < hi && len(words[vt.sorted[i]]) == off {
		vt.nodes[id].term = vt.sorted[i]
		i++
	}
	type group struct {
		r      rune
		lo, hi int32
		size   int
	}
	var groups []group
	for i < hi {
		r, size := utf8.DecodeRuneInString(words[vt.sorted[i]][off:])
		j := i + 1
		for j < hi {
			if r2, _ := utf8.DecodeRuneInString(words[vt.sorted[j]][off:]); r2 != r {
				break
			}
			j++
		}
		groups = append(groups, group{r: r, lo: i, hi: j, size: size})
		i = j
	}
	first := int32(len(vt.edges))
	vt.nodes[id].first, vt.nodes[id].n = first, int32(len(groups))
	for _, g := range groups {
		vt.edges = append(vt.edges, trieEdge{r: g.r})
	}
	for k, g := range groups {
		vt.edges[first+int32(k)].child = vt.build(words, g.lo, g.hi, off+g.size)
	}
	return id
}

// child は節点 n から文字 r の辺をたどった節点を返す。子が少なければ前から、多ければ二分探索で引く。
func (vt *vocabTrie) child(n int32, r rune) (int32, bool) {
	node := vt.nodes[n]
	es := vt.edges[node.first : node.first+node.n]
	if len(es) <= 8 {
		for _, e := range es {
			if e.r == r {
				return e.child, true
			}
		}
		return 0, false
	}
	lo, hi := 0, len(es)
	for lo < hi {
		m := int(uint(lo+hi) >> 1)
		if es[m].r < r {
			lo = m + 1
		} else {
			hi = m
		}
	}
	if lo < len(es) && es[lo].r == r {
		return es[lo].child, true
	}
	return 0, false
}

// walk は s の文字を根から順にたどった節点を返す。
func (vt *vocabTrie) walk(s string) (int32, bool) {
	n := int32(0)
	for _, r := range s {
		c, ok := vt.child(n, r)
		if !ok {
			return 0, false
		}
		n = c
	}
	return n, true
}

// lookup は語の ID を trie で引く。語彙の大きさによらず語の長さに比例する。
func (vt *vocabTrie) lookup(s string) (int32, bool) {
	n, ok := vt.walk(s)
	if !ok || vt.nodes[n].term < 0 {
		return -1, false
	}
	return vt.nodes[n].term, true
}

// prefixRange は prefix で始まる語が並ぶ sorted の範囲を trie で引く。
func (vt *vocabTrie) prefixRange(prefix string) (lo, hi int32) {
	n, ok := vt.walk(prefix)
	if !ok {
		return 0, 0
	}
	return vt.nodes[n].lo, vt.nodes[n].hi
}

// sortedPrefixRange は prefixRange と同じ範囲を、sorted の二分探索で引く。trie と比べるための版。
// prefix で始まる語は prefix 以上の最初の語から連続して並ぶので、その後ろで接頭辞を持たなくなる位置をもう一度二分探索で探す。
func (vt *vocabTrie) sortedPrefixRange(words []string, prefix string) (lo, hi int32) {
	l, _ := slices.BinarySearchFunc(vt.sorted, prefix, func(id int32, p string) int { return strings.Compare(words[id], p) })
	h := l + sort.Search(len(vt.sorted)-l, func(i int) bool { return !strings.HasPrefix(words[vt.sorted[l+i]], prefix) })
	return int32(l), int32(h)
}

// sortedLookup は語の ID を sorted の二分探索で引く。
func (vt *vocabTrie) sortedLookup(words []string, s string) (int32, bool) {
	i, ok := slices.BinarySearchFunc(vt.sorted, s, func(id int32, p string) int { return strings.Compare(words[id], p) })
	if !ok {
		return -1, false
	}
	return vt.sorted[i], true
}

// prefixTerms は prefix で始まる語の ID を、語彙に加えた順（ID の昇順）で返す。
func (vt *vocabTrie) prefixTerms(words []string, prefix string) []int32 {
	lo, hi := vt.prefixRange(prefix)
	out := make([]int32, 0, hi-lo)
	out = append(out, vt.sorted[lo:hi]...)
	for _, id := range vt.invalid {
		if strings.HasPrefix(words[id], prefix) {
			out = append(out, id)
		}
	}
	slices.Sort(out)
	return out
}

type typoMatch struct {
	term  int32
	typos int
}

// typoTerms は typoDistance(q, 語, limit, prefix) が true になる語と打ち間違いの数を、ID の昇順で返す。limit は 1 以上。
// trie を降りながら、制限つきの Damerau-Levenshtein の表を 1 文字ぶんずつ 1 行足す。行の最小値は深くなっても減らないので、上限を超えたら下を見ない。
func (vt *vocabTrie) typoTerms(words []string, q []rune, limit int, prefix bool) []typoMatch {
	w := &typoWalk{vt: vt, q: q, limit: limit, prefix: prefix}
	m := len(q)
	w.rows = make([][]int, m+limit+1)
	w.path = make([]rune, m+limit+1)
	for j := range w.rows {
		w.rows[j] = make([]int, m+1)
	}
	for i := range w.rows[0] {
		w.rows[0][i] = i
	}
	root := vt.nodes[0]
	for k := root.first; k < root.first+root.n; k++ {
		e := vt.edges[k]
		pen := 0
		if e.r != q[0] {
			pen = 1
		}
		w.descend(e.child, e.r, 1, pen, m)
	}
	for _, id := range vt.invalid {
		if d, ok := typoDistance(q, words[id], limit, prefix); ok {
			w.out = append(w.out, typoMatch{term: id, typos: d})
		}
	}
	slices.SortFunc(w.out, func(a, b typoMatch) int { return int(a.term - b.term) })
	return w.out
}

type typoWalk struct {
	vt     *vocabTrie
	q      []rune
	limit  int
	prefix bool
	rows   [][]int
	path   []rune
	out    []typoMatch
}

// descend は深さ j の節点 n（辺の文字 r）の行を求める。pen は最初の文字が違うときに足す 1、best は接頭辞のときの行の末尾のこれまでの最小値。
func (w *typoWalk) descend(n int32, r rune, j, pen, best int) {
	q, m := w.q, len(w.q)
	w.path[j-1] = r
	prev, cur := w.rows[j-1], w.rows[j]
	cur[0] = j
	low := cur[0]
	for i := 1; i <= m; i++ {
		cost := 1
		if q[i-1] == r {
			cost = 0
		}
		cur[i] = min(prev[i]+1, cur[i-1]+1, prev[i-1]+cost)
		if i > 1 && j > 1 && q[i-1] == w.path[j-2] && q[i-2] == r {
			cur[i] = min(cur[i], w.rows[j-2][i-2]+1)
		}
		low = min(low, cur[i])
	}
	node := w.vt.nodes[n]
	if !w.prefix {
		if node.term >= 0 && j >= m-w.limit && cur[m]+pen <= w.limit {
			w.out = append(w.out, typoMatch{term: node.term, typos: cur[m] + pen})
		}
		if j >= m+w.limit || low+pen > w.limit {
			return
		}
		w.children(node, j, pen, best)
		return
	}
	best = min(best, cur[m])
	if j == m+w.limit {
		// これより後ろの文字は typoDistance が切り捨てるので、下の語はすべて同じ距離になる。
		if best+pen <= w.limit {
			for _, id := range w.vt.sorted[node.lo:node.hi] {
				w.out = append(w.out, typoMatch{term: id, typos: best + pen})
			}
		}
		return
	}
	if node.term >= 0 && j >= m-w.limit && best+pen <= w.limit {
		w.out = append(w.out, typoMatch{term: node.term, typos: best + pen})
	}
	if best+pen > w.limit && low+pen > w.limit {
		return
	}
	w.children(node, j, pen, best)
}

func (w *typoWalk) children(node trieNode, j, pen, best int) {
	for k := node.first; k < node.first+node.n; k++ {
		e := w.vt.edges[k]
		w.descend(e.child, e.r, j+1, pen, best)
	}
}
