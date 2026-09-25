package search

import (
	"strings"
	"unicode/utf8"
)

// Meilisearch の既定と同じく、5 文字以上の語は打ち間違いを 1 つ、9 文字以上は 2 つまで許す。
const (
	minLenOneTypo  = 5
	minLenTwoTypos = 9
)

func maxTypos(runes int) int {
	switch {
	case runes >= minLenTwoTypos:
		return 2
	case runes >= minLenOneTypo:
		return 1
	}
	return 0
}

// alternatives はクエリの 1 語に一致させる索引語を返す。先頭は語そのもので、索引に無くても入れる。
// 打ち間違いの許容と接頭辞の一致は英数字の語にだけ使う。日本語の bigram は 2 文字なので許容の対象にならない。
func (ix *Index) alternatives(word string, q Query, last bool, termOf func(string) int) []alt {
	alts := []alt{{term: termOf(word), exact: true}}
	prefix := q.Prefix && last
	if (!q.Typo && !prefix) || strings.ContainsFunc(word, isJapanese) {
		return alts
	}
	limit := 0
	if q.Typo {
		limit = maxTypos(utf8.RuneCountInString(word))
	}
	qr := []rune(word)
	for _, w := range ix.words {
		if w == word {
			continue
		}
		if d, ok := typoDistance(qr, w, limit, prefix); ok {
			alts = append(alts, alt{term: termOf(w), typos: d})
		}
	}
	return alts
}

// typoDistance は q を term（prefix なら term の接頭辞）に変える打ち間違いの数を、置換 / 挿入 / 削除 / 隣り合う 2 文字の入れ替えで数える。
// 最初の文字が違えば 1 つ多く数え、limit を超えたら false を返す。
func typoDistance(q []rune, term string, limit int, prefix bool) (int, bool) {
	if limit == 0 {
		return 0, prefix && strings.HasPrefix(term, string(q))
	}
	n := utf8.RuneCountInString(term)
	if n < len(q)-limit || (!prefix && n > len(q)+limit) {
		return 0, false
	}
	t := []rune(term)
	if prefix {
		t = t[:min(len(t), len(q)+limit)]
	}
	d := osa(q, t, prefix)
	if len(t) > 0 && len(q) > 0 && t[0] != q[0] {
		d++
	}
	return d, d <= limit
}

// osa は制限つきの Damerau-Levenshtein 距離。prefix なら t のどこまでを使うかも選び、最も小さい距離を返す。
func osa(q, t []rune, prefix bool) int {
	prev2 := make([]int, len(t)+1)
	prev := make([]int, len(t)+1)
	cur := make([]int, len(t)+1)
	for j := range prev {
		prev[j] = j
	}
	for i := 1; i <= len(q); i++ {
		cur[0] = i
		for j := 1; j <= len(t); j++ {
			cost := 1
			if q[i-1] == t[j-1] {
				cost = 0
			}
			cur[j] = min(prev[j]+1, cur[j-1]+1, prev[j-1]+cost)
			if i > 1 && j > 1 && q[i-1] == t[j-2] && q[i-2] == t[j-1] {
				cur[j] = min(cur[j], prev2[j-2]+1)
			}
		}
		prev2, prev, cur = prev, cur, prev2
	}
	if !prefix {
		return prev[len(t)]
	}
	best := prev[0]
	for _, d := range prev[1:] {
		best = min(best, d)
	}
	return best
}
