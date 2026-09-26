package morph

import (
	"math"
	"unicode/utf8"
)

const (
	BOS = "BOS"
	EOS = "EOS"
)

// Entry は辞書の 1 つの候補。
type Entry struct {
	Label string
	Cost  float64
}

// Dictionary は表層形から候補の一覧を引く。
type Dictionary map[string][]Entry

// Pair は遷移の前後の品詞。文頭は BOS、文末は EOS。
type Pair struct{ From, To string }

// Transitions は品詞の遷移のコスト。
type Transitions map[Pair]float64

// Node はラティスの 1 つの節。Begin と End は rune の位置。
type Node struct {
	Surface string
	Label   string
	Begin   int
	End     int
	Cost    float64
	Unknown bool
}

// Lattice は入力の文字列の上に辞書の候補を並べたもの。EndAt[p] は位置 p で終わる節の番号。
type Lattice struct {
	Len   int
	Nodes []Node
	EndAt [][]int
}

// Analyzer は辞書と遷移のコストで区切りと品詞を同時に決める。
// 辞書の候補が 1 つも始まらない位置には、UnknownLabel の 1 文字の未知語を置く。
type Analyzer struct {
	Dict         Dictionary
	Trans        Transitions
	DefaultTrans float64
	UnknownLabel string
	UnknownCost  float64
}

func (a Analyzer) transCost(from, to string) float64 {
	if c, ok := a.Trans[Pair{From: from, To: to}]; ok {
		return c
	}
	return a.DefaultTrans
}

// Lattice は text のラティスを作る。
func (a Analyzer) Lattice(text string) Lattice {
	rs := []rune(text)
	maxLen := 0
	for s := range a.Dict {
		maxLen = max(maxLen, utf8.RuneCountInString(s))
	}
	l := Lattice{Len: len(rs), EndAt: make([][]int, len(rs)+1)}
	add := func(n Node) {
		l.EndAt[n.End] = append(l.EndAt[n.End], len(l.Nodes))
		l.Nodes = append(l.Nodes, n)
	}
	for b := range rs {
		found := false
		for e := b + 1; e <= min(len(rs), b+maxLen); e++ {
			s := string(rs[b:e])
			for _, en := range a.Dict[s] {
				add(Node{Surface: s, Label: en.Label, Begin: b, End: e, Cost: en.Cost})
				found = true
			}
		}
		if !found {
			add(Node{Surface: string(rs[b]), Label: a.UnknownLabel, Begin: b, End: b + 1, Cost: a.UnknownCost, Unknown: true})
		}
	}
	return l
}

// Best は BOS から EOS までの最小のコストの経路をビタビで求める。
func (l Lattice) Best(trans func(from, to string) float64) ([]Node, float64) {
	if l.Len == 0 {
		return nil, trans(BOS, EOS)
	}
	cost := make([]float64, len(l.Nodes))
	back := make([]int, len(l.Nodes))
	order := make([]int, 0, len(l.Nodes))
	for p := 1; p <= l.Len; p++ {
		order = append(order, l.EndAt[p]...)
	}
	for _, i := range order {
		n := l.Nodes[i]
		best, arg := math.Inf(1), -1
		if n.Begin == 0 {
			best = trans(BOS, n.Label)
		}
		for _, j := range l.EndAt[n.Begin] {
			if c := cost[j] + trans(l.Nodes[j].Label, n.Label); c < best {
				best, arg = c, j
			}
		}
		cost[i] = best + n.Cost
		back[i] = arg
	}
	best, last := math.Inf(1), -1
	for _, j := range l.EndAt[l.Len] {
		if c := cost[j] + trans(l.Nodes[j].Label, EOS); c < best {
			best, last = c, j
		}
	}
	var path []Node
	for i := last; i >= 0; i = back[i] {
		path = append(path, l.Nodes[i])
	}
	for i, j := 0, len(path)-1; i < j; i, j = i+1, j-1 {
		path[i], path[j] = path[j], path[i]
	}
	return path, best
}

// Analyze は text を区切り、品詞を付けた最小のコストの経路とそのコストを返す。
func (a Analyzer) Analyze(text string) ([]Node, float64) {
	return a.Lattice(text).Best(a.transCost)
}
