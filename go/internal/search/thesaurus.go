package search

import (
	"cmp"
	"slices"
	"strings"
	"sync"
)

// DefaultExpansionLimit は 1 つの語に足す語の上限の既定値。Meilisearch が 1 語の同義語を 50 までに制限するのに合わせる。
const DefaultExpansionLimit = 50

// Thesaurus は同義語のグループと、広い語から狭い語への向きのある関係を持つ類語辞書。
// グループの語はすべて同じ意味として互いに足す。狭い語の関係は広い語に狭い語を足すだけで、狭い語から広い語は足さない（phone に iPhone は足すが、iPhone に phone は足さない）。
// 同じ辞書を Query.Thesaurus（検索のときに広げる）にも WithDocExpansion（索引のときに広げる）にも渡せる。
type Thesaurus struct {
	groups   [][]string
	narrower []narrowerRel
	limit    int
	weight   float64

	mu       sync.Mutex
	compiled map[compileKey]*compiledThesaurus
}

// compileKey の inverse は索引のときに使う向きで、検索のときに X を足す見出し Q について、X を含む文書に Q を足す。
type compileKey struct {
	a       *Analyzer
	inverse bool
}

type narrowerRel struct {
	broader  string
	narrower []string
}

type ThesaurusOption func(*Thesaurus)

// WithSynonymGroup は同じ意味の語を 1 つのグループにする。語の数に上限は無い。
func WithSynonymGroup(words ...string) ThesaurusOption {
	return func(th *Thesaurus) { th.groups = append(th.groups, words) }
}

// WithNarrower は broader に narrower を足す関係を加える。broader の同義語にも同じ narrower を足す。
func WithNarrower(broader string, narrower ...string) ThesaurusOption {
	return func(th *Thesaurus) {
		th.narrower = append(th.narrower, narrowerRel{broader: broader, narrower: narrower})
	}
}

// WithExpansionLimit はクエリや文書の 1 つの語に足す語の数の上限を変える。負の値なら上限を設けない。
func WithExpansionLimit(n int) ThesaurusOption {
	return func(th *Thesaurus) { th.limit = n }
}

// WithExpansionWeight は足した語の点数に掛ける重みを変える。既定は 1 で、同義語を元の語と同じに扱う。
func WithExpansionWeight(w float64) ThesaurusOption {
	return func(th *Thesaurus) { th.weight = w }
}

func NewThesaurus(opts ...ThesaurusOption) *Thesaurus {
	th := &Thesaurus{limit: DefaultExpansionLimit, weight: 1}
	for _, o := range opts {
		o(th)
	}
	return th
}

// compiledThesaurus は見出しを Analyzer で索引語の並びにした辞書。first は見出しの先頭の索引語から、長い順に並べた見出しを引く。
type compiledThesaurus struct {
	first map[string][]*thesaurusEntry
}

// thesaurusEntry の adds は足す語ごとの索引語の並びで、グループの語、狭い語の順に並ぶ。上限はまだ適用していない。
type thesaurusEntry struct {
	key  []string
	adds [][]string
}

// compile は Analyzer と向きごとに 1 回だけ辞書を索引語の並びにする。辞書の語を索引と同じ正規化と分割にそろえないと、クエリや文書の索引語と突き合わせられないため。
// inverse なら検索のときの対応を逆にする。phone のクエリに iPhone を足すのと同じ結果を索引で得るには、iPhone を含む文書に phone を足す。
func (th *Thesaurus) compile(a *Analyzer, inverse bool) *compiledThesaurus {
	th.mu.Lock()
	defer th.mu.Unlock()
	ck := compileKey{a: a, inverse: inverse}
	if c, ok := th.compiled[ck]; ok {
		return c
	}
	type node struct {
		key      []string
		mates    []string
		narrower []string
	}
	var order []string
	nodes := make(map[string]*node)
	get := func(word string) (string, *node) {
		tokens := a.Analyze(word)
		if len(tokens) == 0 {
			return "", nil
		}
		k := strings.Join(tokens, "\x00")
		n, ok := nodes[k]
		if !ok {
			n = &node{key: tokens}
			nodes[k] = n
			order = append(order, k)
		}
		return k, n
	}
	appendNew := func(list []string, k string) []string {
		if slices.Contains(list, k) {
			return list
		}
		return append(list, k)
	}
	for _, g := range th.groups {
		var keys []string
		for _, w := range g {
			if k, n := get(w); n != nil {
				keys = appendNew(keys, k)
			}
		}
		for _, k := range keys {
			for _, m := range keys {
				if m != k {
					nodes[k].mates = appendNew(nodes[k].mates, m)
				}
			}
		}
	}
	for _, r := range th.narrower {
		_, b := get(r.broader)
		if b == nil {
			continue
		}
		for _, w := range r.narrower {
			if k, n := get(w); n != nil {
				b.narrower = appendNew(b.narrower, k)
			}
		}
	}
	adds := make(map[string][]string, len(order))
	for _, k := range order {
		n := nodes[k]
		seen := map[string]bool{k: true}
		push := func(keys []string) {
			for _, a := range keys {
				if !seen[a] {
					seen[a] = true
					adds[k] = append(adds[k], a)
				}
			}
		}
		push(n.mates)
		push(n.narrower)
		for _, m := range n.mates {
			push(nodes[m].narrower)
		}
	}
	if inverse {
		inv := make(map[string][]string, len(order))
		for _, k := range order {
			for _, x := range adds[k] {
				inv[x] = append(inv[x], k)
			}
		}
		adds = inv
	}
	c := &compiledThesaurus{first: make(map[string][]*thesaurusEntry)}
	for _, k := range order {
		if len(adds[k]) == 0 {
			continue
		}
		key := nodes[k].key
		e := &thesaurusEntry{key: key}
		for _, a := range adds[k] {
			e.adds = append(e.adds, nodes[a].key)
		}
		c.first[key[0]] = append(c.first[key[0]], e)
	}
	for _, es := range c.first {
		slices.SortStableFunc(es, func(x, y *thesaurusEntry) int { return cmp.Compare(len(y.key), len(x.key)) })
	}
	if th.compiled == nil {
		th.compiled = make(map[compileKey]*compiledThesaurus)
	}
	th.compiled[ck] = c
	return c
}

// match は words[i:] の先頭に一致する最も長い見出しを返す。
func (c *compiledThesaurus) match(words []string, i int) *thesaurusEntry {
	for _, e := range c.first[words[i]] {
		if len(e.key) <= len(words)-i && slices.Equal(e.key, words[i:i+len(e.key)]) {
			return e
		}
	}
	return nil
}

// capped は上限までの足す語と、上限で捨てた語の数を返す。
func (th *Thesaurus) capped(adds [][]string) ([][]string, int) {
	if th.limit < 0 || len(adds) <= th.limit {
		return adds, 0
	}
	return adds[:th.limit], len(adds) - th.limit
}

// termGroup は類語辞書で広げたクエリの語の並び。covered は元の語の slot、members は足した語ごとの索引語。
type termGroup struct {
	covered []int
	members []groupMember
}

// groupMember の scale は重みに、元の語の数と足した語の索引語の数の比を掛けたもの。bigram に割った長い語が索引語の数だけ高い点を取らないようにする。
type groupMember struct {
	terms []int
	scale float64
}

// expandVariant は書き方の索引語の並びから辞書の見出しを長い順に探し、見出しの slot を termGroup にまとめる。足した語の数と、上限で捨てた語の数を返す。
func (ix *Index) expandVariant(v *variant, tokens []token, th *Thesaurus, termOf func(string) int) (added, dropped int) {
	c := th.compile(ix.analyzer, false)
	words := make([]string, len(tokens))
	for i, t := range tokens {
		words[i] = t.term
	}
	grouped := make([]bool, len(v.slots))
	for i := 0; i < len(words); {
		e := c.match(words, i)
		if e == nil {
			i++
			continue
		}
		var covered []int
		overlap := false
		for k := i; k < i+len(e.key); k++ {
			s := v.seq[k].slot
			overlap = overlap || grouped[s]
			if !slices.Contains(covered, s) {
				covered = append(covered, s)
			}
		}
		if overlap {
			i++
			continue
		}
		adds, d := th.capped(e.adds)
		g := termGroup{covered: covered}
		for _, seq := range adds {
			var ids []int
			for _, w := range seq {
				if id := termOf(w); !slices.Contains(ids, id) {
					ids = append(ids, id)
				}
			}
			g.members = append(g.members, groupMember{terms: ids, scale: th.weight * float64(len(covered)) / float64(len(ids))})
		}
		for _, s := range covered {
			grouped[s] = true
		}
		v.groups = append(v.groups, g)
		added += len(adds)
		dropped += d
		i += len(e.key)
	}
	if len(v.groups) > 0 {
		v.grouped = grouped
	}
	return added, dropped
}

// groupedScore は辞書でまとめていない slot の最大を足し、termGroup ごとに元の語の点数と、足した語ごとの点数のうち最も高いものを足す。
func (v variant) groupedScore(contrib []float64) float64 {
	s := 0.0
	for i, slot := range v.slots {
		if !v.grouped[i] {
			s += slotScore(slot, contrib)
		}
	}
	for _, g := range v.groups {
		best := 0.0
		for _, i := range g.covered {
			best += slotScore(v.slots[i], contrib)
		}
		for _, m := range g.members {
			ms := 0.0
			for _, t := range m.terms {
				ms += contrib[t]
			}
			best = max(best, m.scale*ms)
		}
		s += best
	}
	return s
}

func slotScore(slot []alt, contrib []float64) float64 {
	m := 0.0
	for _, a := range slot {
		m = max(m, contrib[a.term])
	}
	return m
}

// WithDocExpansion は索引を作るときに、文書の語を th で広げた語を同じ位置に重ねて postings に加える（文書拡張）。
// 向きは検索のときの逆で、文書の狭い語に広い語を足す（iPhone を含む文書に phone を足す）。同じ辞書で Query.Thesaurus と同じ文書が一致する。
// 足した語は文書の長さに数えない。Lucene の BM25 が同じ位置に重ねた語を長さから除くのと同じで、足した語の無い文書の点数の正規化を変えないため。
// 出現回数は足した 1 回を th の重み（WithExpansionWeight）倍で数える。複数の索引語から成る語は、見出しの先頭の位置から 1 つずつずらして置く。
// 辞書を変えたら索引を作り直す必要がある。
func WithDocExpansion(th *Thesaurus) Option {
	return func(ix *Index) { ix.docExpansion = th }
}

// weightedToken は索引を作るときに足す語。weight は出現 1 回に掛ける重み。generated は doc2query で生成したクエリの語。
type weightedToken struct {
	tok       token
	weight    float64
	generated bool
}

// expandTokens は文書の 1 項目の索引語の並びから辞書の見出しを長い順に探し、足す語を返す。同じ語を同じ位置に 2 回は足さない。
func (ix *Index) expandTokens(tokens []token) []weightedToken {
	th := ix.docExpansion
	c := th.compile(ix.analyzer, true)
	words := make([]string, len(tokens))
	for i, t := range tokens {
		words[i] = t.term
	}
	type at struct {
		term string
		pos  int32
	}
	var out []weightedToken
	seen := make(map[at]bool)
	for i := 0; i < len(words); {
		e := c.match(words, i)
		if e == nil {
			i++
			continue
		}
		adds, d := th.capped(e.adds)
		ix.expansionDropped += d
		for _, seq := range adds {
			for j, w := range seq {
				p := at{term: w, pos: tokens[i].pos + int32(j)}
				if seen[p] {
					continue
				}
				seen[p] = true
				out = append(out, weightedToken{tok: token{term: w, pos: p.pos}, weight: th.weight})
			}
		}
		i += len(e.key)
	}
	return out
}

// mergeTokens は位置の昇順に並べ直す。同じ位置では元の語を先に置く。
func mergeTokens(tokens []token, extra []weightedToken) []token {
	out := make([]token, 0, len(tokens)+len(extra))
	out = append(out, tokens...)
	for _, e := range extra {
		out = append(out, e.tok)
	}
	slices.SortStableFunc(out, func(a, b token) int { return cmp.Compare(a.pos, b.pos) })
	return out
}
