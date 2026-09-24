package search

import (
	"cmp"
	"slices"
	"strings"
)

// maxVariants は書き方の集合を uint64 のビットで持つための上限。
const maxVariants = 64

type queryPos struct {
	term int
	off  int32
}

// variant はクエリの 1 つの書き方。terms は重複を除いた語、seq はフレーズの照合に使う語の並びと先頭からの間隔。
type variant struct {
	terms []int
	seq   []queryPos
}

// parsedQuery の terms は語彙の ID。索引に無い語は -1 にする。words は同じ並びの索引語で、シャードをまたいで統計を突き合わせるのに使う。
type parsedQuery struct {
	terms    []int32
	words    []string
	variants []variant
}

func (ix *Index) parse(q Query) parsedQuery {
	var pq parsedQuery
	slot := make(map[string]int)
	for _, text := range expandSynonyms(ix.analyzer.normalizeText(q.Text), q.Synonyms, ix.analyzer.normalizeText) {
		tokens := ix.analyzer.analyzeNormalized(text)
		var v variant
		for _, t := range tokens {
			i, ok := slot[t.term]
			if !ok {
				i = len(pq.terms)
				slot[t.term] = i
				pq.terms = append(pq.terms, ix.lookupTerm(t.term))
				pq.words = append(pq.words, t.term)
			}
			if !slices.Contains(v.terms, i) {
				v.terms = append(v.terms, i)
			}
			v.seq = append(v.seq, queryPos{term: i, off: t.pos - tokens[0].pos})
		}
		pq.variants = append(pq.variants, v)
	}
	return pq
}

// expandSynonyms は辞書の見出しか置き換え先を含むクエリに、もう一方で書いた版を足す。索引は変えないので、辞書を変えても作り直さずに済む。
func expandSynonyms(text string, dict map[string]string, normalize func(string) string) []string {
	texts := []string{text}
	if len(dict) == 0 {
		return texts
	}
	pairs := make([][2]string, 0, len(dict))
	for from, to := range dict {
		from, to = normalize(from), normalize(to)
		if from != "" && to != "" && from != to {
			pairs = append(pairs, [2]string{from, to})
		}
	}
	slices.SortFunc(pairs, func(x, y [2]string) int { return cmp.Or(cmp.Compare(x[0], y[0]), cmp.Compare(x[1], y[1])) })
	for _, p := range pairs {
		for _, t := range texts {
			for _, r := range [][2]string{p, {p[1], p[0]}} {
				if !strings.Contains(t, r[0]) {
					continue
				}
				if w := strings.ReplaceAll(t, r[0], r[1]); !slices.Contains(texts, w) && len(texts) < maxVariants {
					texts = append(texts, w)
				}
			}
		}
	}
	return texts
}

func (ix *Index) lookupTerm(term string) int32 {
	if id, ok := ix.vocab[term]; ok {
		return id
	}
	return -1
}
