// Package fst は、辞書順に並んだ語と値の組から最小の非巡回の有限状態トランスデューサ（FST）を作る。
// 先頭だけでなく末尾が共通する語も同じ状態を共有し、辺の出力の和で語ごとの値を区別する。
package fst

import (
	"errors"
	"slices"
	"strconv"
	"strings"
)

// ErrOrder は Add に渡した語が、直前の語より辞書順で後ろにないときに返す。
var ErrOrder = errors.New("fst: keys must be added in strictly increasing order")

type trans struct {
	label rune
	out   uint64
	to    int
}

type node struct {
	final    bool
	finalOut uint64
	trans    []trans
}

// FST は作った後は読むだけなので、複数の goroutine から同時に使える。
type FST struct {
	nodes []node
	root  int
	keys  int
}

// Builder は語を辞書順に受け取り、確定した部分から同じ形の状態をまとめていく（Daciuk らの逐次的な最小化）。
type Builder struct {
	stack    []*node
	prev     []rune
	registry map[string]int
	nodes    []node
	keys     int
	started  bool
}

func NewBuilder() *Builder {
	return &Builder{stack: []*node{{}}, registry: make(map[string]int)}
}

// Add は key と値 out を足す。key は直前の語より辞書順（rune の並び）で後ろでなければならない。
func (b *Builder) Add(key string, out uint64) error {
	k := []rune(key)
	if b.started && slices.Compare(b.prev, k) >= 0 {
		return ErrOrder
	}
	p := commonPrefix(b.prev, k)
	b.freezeFrom(p)
	// 共通の接頭辞の辺には両方の値の小さいほうだけを残し、余りは 1 つ先の状態から出る辺と終了の出力へ押し出す。
	for i := range p {
		t := &b.stack[i].trans[len(b.stack[i].trans)-1]
		common := min(t.out, out)
		if rest := t.out - common; rest > 0 {
			next := b.stack[i+1]
			for j := range next.trans {
				next.trans[j].out += rest
			}
			if next.final {
				next.finalOut += rest
			}
		}
		t.out = common
		out -= common
	}
	// 余りの押し出しは既にある辺にだけかけるので、新しい語の辺はその後に足す。
	for i := p; i < len(k); i++ {
		b.stack[i].trans = append(b.stack[i].trans, trans{label: k[i], to: -1})
		b.stack = append(b.stack, &node{})
	}
	if p < len(k) {
		b.stack[p].trans[len(b.stack[p].trans)-1].out = out
	} else {
		b.stack[p].finalOut = out
	}
	b.stack[len(k)].final = true
	b.prev = k
	b.started = true
	b.keys++
	return nil
}

// Finish は残りの状態を確定して FST を返す。Builder はそれ以降使えない。
func (b *Builder) Finish() *FST {
	b.freezeFrom(0)
	root := b.freeze(b.stack[0])
	return &FST{nodes: b.nodes, root: root, keys: b.keys}
}

// freezeFrom は直前の語のうち、位置 p より深い状態を確定し、親の最後の辺をその確定した状態につなぐ。
func (b *Builder) freezeFrom(p int) {
	for i := len(b.stack) - 1; i > p; i-- {
		id := b.freeze(b.stack[i])
		parent := b.stack[i-1]
		parent.trans[len(parent.trans)-1].to = id
	}
	b.stack = b.stack[:p+1]
}

func (b *Builder) freeze(n *node) int {
	sig := signature(n)
	if id, ok := b.registry[sig]; ok {
		return id
	}
	b.nodes = append(b.nodes, node{final: n.final, finalOut: n.finalOut, trans: slices.Clone(n.trans)})
	id := len(b.nodes) - 1
	b.registry[sig] = id
	return id
}

func signature(n *node) string {
	var s strings.Builder
	if n.final {
		s.WriteString("F")
		s.WriteString(strconv.FormatUint(n.finalOut, 36))
	}
	for _, t := range n.trans {
		s.WriteByte('|')
		s.WriteString(strconv.FormatInt(int64(t.label), 36))
		s.WriteByte(',')
		s.WriteString(strconv.FormatUint(t.out, 36))
		s.WriteByte(',')
		s.WriteString(strconv.Itoa(t.to))
	}
	return s.String()
}

func commonPrefix(a, b []rune) int {
	n := min(len(a), len(b))
	for i := range n {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}

// Build は辞書順に並んだ keys と、それぞれの値 outs から FST を作る。
func Build(keys []string, outs []uint64) (*FST, error) {
	b := NewBuilder()
	for i, k := range keys {
		if err := b.Add(k, outs[i]); err != nil {
			return nil, err
		}
	}
	return b.Finish(), nil
}

// Get は key の値を返す。途中の辺の出力と終了の出力の和が、Add で渡した値になる。
func (f *FST) Get(key string) (uint64, bool) {
	s, sum := f.root, uint64(0)
	for _, r := range key {
		t, ok := f.next(s, r)
		if !ok {
			return 0, false
		}
		sum += t.out
		s = t.to
	}
	n := &f.nodes[s]
	if !n.final {
		return 0, false
	}
	return sum + n.finalOut, true
}

func (f *FST) next(s int, r rune) (trans, bool) {
	ts := f.nodes[s].trans
	i, ok := slices.BinarySearchFunc(ts, r, func(t trans, r rune) int { return int(t.label - r) })
	if !ok {
		return trans{}, false
	}
	return ts[i], true
}

// Entry は FST から取り出した語と値。
type Entry struct {
	Key string
	Out uint64
}

// Prefix は prefix で始まる語を辞書順に返す。limit が 0 以下なら全部。
func (f *FST) Prefix(prefix string, limit int) []Entry {
	s, sum := f.root, uint64(0)
	for _, r := range prefix {
		t, ok := f.next(s, r)
		if !ok {
			return nil
		}
		sum += t.out
		s = t.to
	}
	var out []Entry
	buf := []rune(prefix)
	var walk func(s int, sum uint64) bool
	walk = func(s int, sum uint64) bool {
		n := &f.nodes[s]
		if n.final {
			out = append(out, Entry{Key: string(buf), Out: sum + n.finalOut})
			if limit > 0 && len(out) >= limit {
				return false
			}
		}
		for _, t := range n.trans {
			buf = append(buf, t.label)
			ok := walk(t.to, sum+t.out)
			buf = buf[:len(buf)-1]
			if !ok {
				return false
			}
		}
		return true
	}
	walk(s, sum)
	return out
}

// Len は語の数を返す。
func (f *FST) Len() int { return f.keys }

// States は状態の数、Transitions は辺の数を返す。トライ木と比べて共有の効果を見るために使う。
func (f *FST) States() int { return len(f.nodes) }

func (f *FST) Transitions() int {
	n := 0
	for i := range f.nodes {
		n += len(f.nodes[i].trans)
	}
	return n
}
