package search

import (
	"cmp"
	"errors"
	"fmt"
	"slices"
	"strings"
	"unicode"
)

// ErrExprSyntax は論理式の構文の誤り。ParseExpr の返すエラーはこれを包む。
var ErrExprSyntax = errors.New("search: expression syntax")

type exprOp int

const (
	exprTerm exprOp = iota
	exprAnd
	exprOr
)

// Expr は語を AND / OR / 括弧で組んだ論理式。AND は OR より強く結び付く。ParseExpr で作る。
type Expr struct {
	op   exprOp
	text string
	args []*Expr
}

// ParseExpr は「(カツ OR カレー) AND おいしい」のような式を読む。演算子は大文字の AND / OR で、括弧は全角も受け付ける。語を演算子なしで並べるとエラーにする。
func ParseExpr(s string) (*Expr, error) {
	p := exprParser{tokens: lexExpr(s)}
	if len(p.tokens) == 0 {
		return nil, fmt.Errorf("%w: empty expression", ErrExprSyntax)
	}
	e, err := p.or()
	if err != nil {
		return nil, err
	}
	if p.i < len(p.tokens) {
		return nil, fmt.Errorf("%w: unexpected %q", ErrExprSyntax, p.tokens[p.i])
	}
	return e, nil
}

// Words は式に現れる語を、左から現れた順に返す。
func (e *Expr) Words() []string {
	if e.op == exprTerm {
		return []string{e.text}
	}
	var out []string
	for _, a := range e.args {
		out = append(out, a.Words()...)
	}
	return out
}

func (e *Expr) String() string {
	if e.op == exprTerm {
		return e.text
	}
	sep := " AND "
	if e.op == exprOr {
		sep = " OR "
	}
	parts := make([]string, len(e.args))
	for i, a := range e.args {
		parts[i] = a.String()
	}
	return "(" + strings.Join(parts, sep) + ")"
}

func lexExpr(s string) []string {
	var tokens []string
	var word strings.Builder
	flush := func() {
		if word.Len() > 0 {
			tokens = append(tokens, word.String())
			word.Reset()
		}
	}
	for _, r := range s {
		switch {
		case r == '(' || r == '（':
			flush()
			tokens = append(tokens, "(")
		case r == ')' || r == '）':
			flush()
			tokens = append(tokens, ")")
		case unicode.IsSpace(r):
			flush()
		default:
			word.WriteRune(r)
		}
	}
	flush()
	return tokens
}

type exprParser struct {
	tokens []string
	i      int
}

func (p *exprParser) peek() string {
	if p.i < len(p.tokens) {
		return p.tokens[p.i]
	}
	return ""
}

func (p *exprParser) or() (*Expr, error) {
	return p.chain("OR", exprOr, p.and)
}

func (p *exprParser) and() (*Expr, error) {
	return p.chain("AND", exprAnd, p.primary)
}

func (p *exprParser) chain(keyword string, op exprOp, next func() (*Expr, error)) (*Expr, error) {
	first, err := next()
	if err != nil {
		return nil, err
	}
	args := []*Expr{first}
	for p.peek() == keyword {
		p.i++
		e, err := next()
		if err != nil {
			return nil, err
		}
		args = append(args, e)
	}
	if len(args) == 1 {
		return first, nil
	}
	return &Expr{op: op, args: args}, nil
}

func (p *exprParser) primary() (*Expr, error) {
	switch t := p.peek(); t {
	case "":
		return nil, fmt.Errorf("%w: missing term at end", ErrExprSyntax)
	case "(":
		p.i++
		e, err := p.or()
		if err != nil {
			return nil, err
		}
		if p.peek() != ")" {
			return nil, fmt.Errorf("%w: missing )", ErrExprSyntax)
		}
		p.i++
		return e, nil
	case ")", "AND", "OR":
		return nil, fmt.Errorf("%w: unexpected %q", ErrExprSyntax, t)
	default:
		p.i++
		if n := p.peek(); n != "" && n != ")" && n != "AND" && n != "OR" {
			return nil, fmt.Errorf("%w: missing operator between %q and %q", ErrExprSyntax, t, n)
		}
		return &Expr{op: exprTerm, text: t}, nil
	}
}

// scoringText は Expr があれば、式の語すべてを並べた文字列を採点に使う Text にする。
func (q Query) scoringText() Query {
	if q.Expr != nil {
		q.Text = strings.Join(q.Expr.Words(), " ")
	}
	return q
}

// docList は式の途中の結果。leaf なら索引の postings をそのまま指し、項目で絞る前の状態で持つ。そうでなければ ids が絞った後の文書番号で、書き換えてよい。
type docList struct {
	ids  []int32
	ps   []posting
	leaf bool
}

func (l docList) size() int {
	if l.leaf {
		return len(l.ps)
	}
	return len(l.ids)
}

func (l docList) docs(sel fieldSet, all bool) []int32 {
	if l.leaf {
		return docsOf(l.ps, sel, all)
	}
	return l.ids
}

// matchExpr は式を満たす文書の番号を昇順で返す。
func (ix *Index) matchExpr(e *Expr, sel fieldSet) []int {
	all := sel == selectFields(nil)
	ids := ix.evalExpr(e, sel, all).docs(sel, all)
	out := make([]int, len(ids))
	for i, id := range ids {
		out[i] = int(id)
	}
	return out
}

// evalExpr の語は Analyzer で索引語に分け、索引語が複数ならすべてを含む文書にする。
func (ix *Index) evalExpr(e *Expr, sel fieldSet, all bool) docList {
	switch e.op {
	case exprOr:
		ids := ix.evalExpr(e.args[0], sel, all).docs(sel, all)
		for _, a := range e.args[1:] {
			ids = unionIDs(ids, ix.evalExpr(a, sel, all).docs(sel, all))
		}
		return docList{ids: ids}
	case exprAnd:
		kids := make([]docList, len(e.args))
		for i, a := range e.args {
			kids[i] = ix.evalExpr(a, sel, all)
		}
		slices.SortFunc(kids, func(a, b docList) int { return cmp.Compare(a.size(), b.size()) })
		ids := kids[0].docs(sel, all)
		for _, k := range kids[1:] {
			if len(ids) == 0 {
				break
			}
			if k.leaf {
				ids = intersect(ids, k.ps, sel, all, ix.gallopRatio)
			} else {
				ids = intersectIDs(ids, k.ids, ix.gallopRatio)
			}
		}
		return docList{ids: ids}
	default:
		var lists [][]posting
		seen := make(map[int32]bool)
		for _, t := range ix.analyzer.analyze(e.text) {
			id := ix.lookupTerm(t.term)
			if !seen[id] {
				seen[id] = true
				lists = append(lists, ix.postingsOf(id))
			}
		}
		switch len(lists) {
		case 0:
			return docList{}
		case 1:
			return docList{ps: lists[0], leaf: true}
		}
		return docList{ids: intersectAll(lists, sel, all, ix.gallopRatio, nil)}
	}
}

// unionIDs は昇順の 2 つの列を前から併合し、重複を 1 つにした新しい列を返す。
func unionIDs(a, b []int32) []int32 {
	out := make([]int32, 0, len(a)+len(b))
	i, j := 0, 0
	for i < len(a) && j < len(b) {
		switch x, y := a[i], b[j]; {
		case x < y:
			out = append(out, x)
			i++
		case x > y:
			out = append(out, y)
			j++
		default:
			out = append(out, x)
			i++
			j++
		}
	}
	out = append(out, a[i:]...)
	return append(out, b[j:]...)
}

// intersectIDs は途中の結果どうしの共通部分を ids の領域に書いて返す。長さの比で 2 つのポインタと galloping を切り替えるのは intersect と同じ。
func intersectIDs(ids, other []int32, ratio int) []int32 {
	out := ids[:0]
	if useGallop(len(ids), len(other), ratio) {
		j := 0
		for _, a := range ids {
			if j >= len(other) {
				break
			}
			lo, step := j, 1
			for lo+step < len(other) && other[lo+step] < a {
				lo += step
				step *= 2
			}
			k, found := slices.BinarySearch(other[lo:min(lo+step+1, len(other))], a)
			j = lo + k
			if found {
				out = append(out, a)
				j++
			}
		}
		return out
	}
	i, j := 0, 0
	for i < len(ids) && j < len(other) {
		switch a, b := ids[i], other[j]; {
		case a < b:
			i++
		case a > b:
			j++
		default:
			out = append(out, a)
			i++
			j++
		}
	}
	return out
}
