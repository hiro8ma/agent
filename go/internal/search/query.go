package search

import (
	"cmp"
	"slices"
	"strings"
)

// maxVariants は書き方の集合を uint64 のビットで持つための上限。
const maxVariants = 64

// queryPos の slot は variant.slots の添字。
type queryPos struct {
	slot int
	off  int32
}

// alt はクエリの 1 語に一致させる索引語の 1 つ。term は parsedQuery.terms の添字。exact はクエリの語そのもので、打ち間違いも接頭辞の補完も無い。
type alt struct {
	term  int
	typos int
	exact bool
}

// variant はクエリの 1 つの書き方。slots は重複を除いたクエリの語ごとの一致させる索引語、seq はフレーズの照合に使う語の並びと先頭からの間隔。
type variant struct {
	slots [][]alt
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
	index := make(map[string]int)
	termOf := func(word string) int {
		i, ok := index[word]
		if !ok {
			i = len(pq.terms)
			index[word] = i
			pq.terms = append(pq.terms, ix.lookupTerm(word))
			pq.words = append(pq.words, word)
		}
		return i
	}
	for _, text := range expandSynonyms(ix.analyzer.normalizeText(q.Text), q.Synonyms, ix.analyzer.normalizeText) {
		tokens := ix.analyzer.analyzeNormalized(text)
		var v variant
		slotOf := make(map[string]int)
		for k, t := range tokens {
			si, ok := slotOf[t.term]
			if !ok {
				si = len(v.slots)
				slotOf[t.term] = si
				v.slots = append(v.slots, ix.alternatives(t.term, q, k == len(tokens)-1, termOf))
			}
			v.seq = append(v.seq, queryPos{slot: si, off: t.pos - tokens[0].pos})
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
