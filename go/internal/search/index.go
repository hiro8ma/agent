// Package search は転置インデックスで候補を絞り、候補だけを BM25 で採点する全文検索。
package search

import (
	"cmp"
	"context"
	"math"
	"slices"
)

const (
	DefaultK1 = 1.2
	DefaultB  = 0.75
)

// Doc は検索の対象の 1 文書。タイトルと本文をまとめて語に分ける。
type Doc struct {
	Title   string
	Content string
}

type posting struct {
	doc int
	tf  int
}

type Index struct {
	k1, b    float64
	docs     []Doc
	lengths  []int
	avgLen   float64
	postings map[string][]posting
}

type Option func(*Index)

func WithK1(k1 float64) Option { return func(ix *Index) { ix.k1 = k1 } }

func WithB(b float64) Option { return func(ix *Index) { ix.b = b } }

// New は文書を 1 回だけ走査して転置インデックスを作る。postings は文書 ID の昇順に並ぶ。
func New(docs []Doc, opts ...Option) *Index {
	ix := &Index{
		k1:       DefaultK1,
		b:        DefaultB,
		docs:     docs,
		lengths:  make([]int, len(docs)),
		postings: make(map[string][]posting),
	}
	for _, o := range opts {
		o(ix)
	}
	total := 0
	for id, d := range docs {
		tokens := Tokenize(d.Title + " " + d.Content)
		ix.lengths[id] = len(tokens)
		total += len(tokens)
		tf := make(map[string]int)
		for _, t := range tokens {
			tf[t]++
		}
		for t, n := range tf {
			ix.postings[t] = append(ix.postings[t], posting{doc: id, tf: n})
		}
	}
	if len(docs) > 0 {
		ix.avgLen = float64(total) / float64(len(docs))
	}
	return ix
}

type Hit struct {
	Doc   Doc
	Score float64
}

type Result struct {
	Hits   []Hit
	Scored int
}

func (ix *Index) Search(ctx context.Context, query string, limit int) ([]Doc, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	res := ix.Rank(query, limit)
	docs := make([]Doc, len(res.Hits))
	for i, h := range res.Hits {
		docs[i] = h.Doc
	}
	return docs, nil
}

func (ix *Index) Rank(query string, limit int) Result {
	terms := queryTerms(query)
	candidates := ix.match(terms)
	hits := make([]Hit, 0, len(candidates))
	for _, id := range candidates {
		hits = append(hits, Hit{Doc: ix.docs[id], Score: ix.score(id, terms)})
	}
	slices.SortStableFunc(hits, func(a, b Hit) int { return cmp.Compare(b.Score, a.Score) })
	if limit >= 0 && len(hits) > limit {
		hits = hits[:limit]
	}
	return Result{Hits: hits, Scored: len(candidates)}
}

// match はクエリ語の postings の和集合を返す。どの語も含まない文書はここで落ち、採点されない。
func (ix *Index) match(terms []string) []int {
	seen := make(map[int]struct{})
	var ids []int
	for _, t := range terms {
		for _, p := range ix.postings[t] {
			if _, ok := seen[p.doc]; !ok {
				seen[p.doc] = struct{}{}
				ids = append(ids, p.doc)
			}
		}
	}
	slices.Sort(ids)
	return ids
}

func (ix *Index) score(id int, terms []string) float64 {
	norm := 1 - ix.b + ix.b*float64(ix.lengths[id])/ix.avgLen
	s := 0.0
	for _, t := range terms {
		ps := ix.postings[t]
		i, ok := slices.BinarySearchFunc(ps, id, func(p posting, id int) int { return cmp.Compare(p.doc, id) })
		if !ok {
			continue
		}
		tf := float64(ps[i].tf)
		s += ix.idf(len(ps)) * tf * (ix.k1 + 1) / (tf + ix.k1*norm)
	}
	return s
}

func (ix *Index) idf(df int) float64 {
	n := float64(len(ix.docs))
	return math.Log(1 + (n-float64(df)+0.5)/(float64(df)+0.5))
}

func queryTerms(query string) []string {
	var terms []string
	seen := make(map[string]struct{})
	for _, t := range Tokenize(query) {
		if _, ok := seen[t]; !ok {
			seen[t] = struct{}{}
			terms = append(terms, t)
		}
	}
	return terms
}
