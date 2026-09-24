package search

import (
	"cmp"
	"maps"
	"slices"
	"strings"
	"unicode"
	"unicode/utf8"

	"golang.org/x/text/unicode/norm"
)

// Analyzer は文字列を索引語に変える。正規化、辞書の置き換え、トークン分割、除去の順に処理し、文書とクエリに同じものを使う。
type Analyzer struct {
	normalize bool
	synonyms  *strings.Replacer
	stop      map[string]struct{}
}

type analyzerConfig struct {
	normalize bool
	synonyms  map[string]string
	stop      []string
}

type AnalyzerOption func(*analyzerConfig)

// WithNormalization は NFKC、英字の小文字化、カタカナの長音の揺れの統一を有効にする。
func WithNormalization() AnalyzerOption {
	return func(c *analyzerConfig) { c.normalize = true }
}

// WithSynonyms は表記ゆれや同義語を、トークン分割の前に文字列のまま置き換える。bigram に割ってからでは語の対応が取れないため。
func WithSynonyms(dict map[string]string) AnalyzerOption {
	return func(c *analyzerConfig) {
		if c.synonyms == nil {
			c.synonyms = make(map[string]string, len(dict))
		}
		maps.Copy(c.synonyms, dict)
	}
}

func WithStopWords(words ...string) AnalyzerOption {
	return func(c *analyzerConfig) { c.stop = append(c.stop, words...) }
}

var (
	japaneseParticleStopWords = []string{
		"の", "は", "が", "を", "に", "で", "と", "も", "へ", "や",
		"には", "では", "とは", "との", "での", "への", "にも", "でも",
		"のは", "のが", "のを", "のに", "ので", "のも",
	}
	englishStopWords = []string{
		"a", "an", "and", "are", "as", "at", "be", "by", "for", "in",
		"is", "it", "of", "on", "or", "the", "to", "with",
	}
)

// DefaultStopWords は助詞だけから成る日本語の unigram と bigram、Lucene の英語の除去語の一部を返す。
func DefaultStopWords() []string {
	return slices.Concat(japaneseParticleStopWords, englishStopWords)
}

// NewAnalyzer は既定では Tokenize だけをかける。
func NewAnalyzer(opts ...AnalyzerOption) *Analyzer {
	var c analyzerConfig
	for _, o := range opts {
		o(&c)
	}
	a := &Analyzer{normalize: c.normalize}
	if len(c.synonyms) > 0 {
		pairs := make([][2]string, 0, len(c.synonyms))
		for from, to := range c.synonyms {
			from, to = a.normalizeText(from), a.normalizeText(to)
			if from != "" && from != to {
				pairs = append(pairs, [2]string{from, to})
			}
		}
		// strings.Replacer は同じ位置で引数の順に照合するので、長い見出しを先に置いて最長一致にする。
		slices.SortFunc(pairs, func(x, y [2]string) int {
			return cmp.Or(cmp.Compare(utf8.RuneCountInString(y[0]), utf8.RuneCountInString(x[0])), cmp.Compare(x[0], y[0]))
		})
		oldnew := make([]string, 0, 2*len(pairs))
		for _, p := range pairs {
			oldnew = append(oldnew, p[0], p[1])
		}
		if len(oldnew) > 0 {
			a.synonyms = strings.NewReplacer(oldnew...)
		}
	}
	if len(c.stop) > 0 {
		a.stop = make(map[string]struct{}, len(c.stop))
		for _, w := range c.stop {
			a.stop[strings.ToLower(a.normalizeText(w))] = struct{}{}
		}
	}
	return a
}

func (a *Analyzer) Analyze(text string) []string {
	tokens := a.analyze(text)
	terms := make([]string, len(tokens))
	for i, t := range tokens {
		terms[i] = t.term
	}
	return terms
}

// token の pos は除去の前の並びでの位置。除去した語の分だけ位置が空くので、フレーズの照合で語の間隔を保てる。
type token struct {
	term string
	pos  int32
}

func (a *Analyzer) analyze(text string) []token {
	return a.analyzeNormalized(a.normalizeText(text))
}

func (a *Analyzer) analyzeNormalized(text string) []token {
	if a.synonyms != nil {
		text = a.synonyms.Replace(text)
	}
	terms := Tokenize(text)
	tokens := make([]token, 0, len(terms))
	for i, t := range terms {
		if _, ok := a.stop[t]; ok {
			continue
		}
		tokens = append(tokens, token{term: t, pos: int32(i)})
	}
	return tokens
}

func (a *Analyzer) normalizeText(s string) string {
	if !a.normalize {
		return s
	}
	return Normalize(s)
}

// kuromoji の stemmer と同じく、4 文字以上のカタカナ語に限って末尾の長音を落とす（コンピューター と コンピュータ）。
const minLongVowelStemLen = 4

// Normalize は NFKC で全角英数字と半角カナを揃え、英字を小文字にし、カタカナの直後のダッシュ類を長音にしてから、
// 4 文字以上のカタカナ語の末尾の長音を落とす。
func Normalize(s string) string {
	s = strings.ToLower(norm.NFKC.String(s))
	var (
		b   strings.Builder
		run []rune
	)
	b.Grow(len(s))
	flush := func() {
		if len(run) >= minLongVowelStemLen && run[len(run)-1] == 'ー' {
			run = run[:len(run)-1]
		}
		b.WriteString(string(run))
		run = run[:0]
	}
	for _, r := range s {
		if len(run) > 0 && isDashLike(r) {
			r = 'ー'
		}
		if isKatakana(r) {
			run = append(run, r)
			continue
		}
		flush()
		b.WriteRune(r)
	}
	flush()
	return b.String()
}

func isKatakana(r rune) bool {
	return r == 'ー' || unicode.Is(unicode.Katakana, r)
}

// isDashLike は PDF や入力の誤りで長音の代わりに入るダッシュ類。複合語の区切りに使う ASCII のハイフンは含めない。
func isDashLike(r rune) bool {
	switch r {
	case '‐', '‑', '–', '—', '―', '−', '─', '━':
		return true
	}
	return false
}
