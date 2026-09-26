package search

import (
	"cmp"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
)

var contractions = map[string]string{
	"i'm": "i am", "you're": "you are", "we're": "we are", "they're": "they are",
	"it's": "it is", "he's": "he is", "she's": "she is", "that's": "that is",
	"there's": "there is", "let's": "let us",
	"i've": "i have", "you've": "you have", "we've": "we have", "they've": "they have",
	"i'll": "i will", "you'll": "you will", "we'll": "we will", "they'll": "they will",
	"i'd": "i would", "you'd": "you would",
	"don't": "do not", "doesn't": "does not", "didn't": "did not",
	"can't": "can not", "won't": "will not", "isn't": "is not", "aren't": "are not",
	"wasn't": "was not", "weren't": "were not", "haven't": "have not", "hasn't": "has not",
}

var elisions = []string{"qu'", "l'", "d'", "j'", "m'", "n'", "s'", "t'", "c'"}

var (
	ipv4Re    = regexp.MustCompile(`^\d{1,3}(?:\.\d{1,3}){3}`)
	acronymRe = regexp.MustCompile(`^[A-Za-z](?:\.[A-Za-z])+\.?`)
	domainRe  = regexp.MustCompile(`^[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)*\.[A-Za-z]{2,}`)
)

type englishTokenizer struct {
	apostrophes bool
	symbols     bool
	phrases     map[string][][]string
}

func newEnglishTokenizer(english bool, phrases []string) *englishTokenizer {
	et := &englishTokenizer{apostrophes: english, symbols: english}
	for _, p := range phrases {
		ws := strings.Fields(strings.ToLower(p))
		if len(ws) < 2 {
			continue
		}
		if et.phrases == nil {
			et.phrases = make(map[string][][]string)
		}
		et.phrases[ws[0]] = append(et.phrases[ws[0]], ws)
	}
	for _, cands := range et.phrases {
		slices.SortFunc(cands, func(x, y []string) int { return cmp.Compare(len(y), len(x)) })
	}
	return et
}

// englishWord の joined は直前の語との間が空白かハイフンだけのとき true。複数語の辞書はこの間でだけつなぐ。
type englishWord struct {
	term   string
	latin  bool
	joined bool
}

func (et *englishTokenizer) tokenize(text string) []string {
	words := et.split(text)
	tokens := make([]string, 0, len(words))
	for i := 0; i < len(words); i++ {
		if n := et.matchPhrase(words[i:]); n > 0 {
			terms := make([]string, n)
			for k := range n {
				terms[k] = words[i+k].term
			}
			tokens = append(tokens, strings.Join(terms, "_"))
			i += n - 1
			continue
		}
		tokens = append(tokens, words[i].term)
	}
	return tokens
}

func (et *englishTokenizer) matchPhrase(words []englishWord) int {
	if !words[0].latin {
		return 0
	}
	for _, p := range et.phrases[words[0].term] {
		if len(p) > len(words) {
			continue
		}
		ok := true
		for k := 1; k < len(p) && ok; k++ {
			ok = words[k].latin && words[k].joined && words[k].term == p[k]
		}
		if ok {
			return len(p)
		}
	}
	return 0
}

func (et *englishTokenizer) split(text string) []englishWord {
	var (
		words     []englishWord
		word      strings.Builder
		ja        []rune
		gapOK     bool
		prevLatin bool
	)
	emit := func(w englishWord) {
		w.joined = w.latin && prevLatin && gapOK
		words = append(words, w)
		prevLatin, gapOK = w.latin, true
	}
	flushWord := func() {
		if word.Len() == 0 {
			return
		}
		w := word.String()
		word.Reset()
		if !strings.Contains(w, "'") {
			emit(englishWord{term: w, latin: true})
			return
		}
		for _, t := range expandApostrophe(w) {
			emit(englishWord{term: t, latin: true})
		}
	}
	flushJa := func() {
		if len(ja) == 0 {
			return
		}
		if len(ja) == 1 {
			emit(englishWord{term: string(ja)})
		}
		for i := range len(ja) - 1 {
			emit(englishWord{term: string(ja[i : i+2])})
		}
		ja = ja[:0]
	}
	for i := 0; i < len(text); {
		r, size := utf8.DecodeRuneInString(text[i:])
		switch {
		case isJapanese(r):
			flushWord()
			ja = append(ja, r)
		case unicode.IsLetter(r) || unicode.IsDigit(r):
			flushJa()
			if word.Len() == 0 && et.symbols && !continuesChunk(text[:i]) {
				if term, n := matchSymbol(text[i:]); n > 0 {
					emit(englishWord{term: term})
					i += n
					continue
				}
			}
			word.WriteRune(unicode.ToLower(r))
		case et.apostrophes && isApostrophe(r) && word.Len() > 0 && startsWithLetter(text[i+size:]):
			word.WriteByte('\'')
		default:
			flushWord()
			flushJa()
			if !unicode.IsSpace(r) && r != '-' {
				gapOK = false
			}
		}
		i += size
	}
	flushWord()
	flushJa()
	return words
}

func isApostrophe(r rune) bool { return r == '\'' || r == '’' }

func startsWithLetter(s string) bool {
	r, _ := utf8.DecodeRuneInString(s)
	return unicode.IsLetter(r)
}

// expandApostrophe は短縮形を辞書で展開し、先頭のエリジオン（l'）と末尾の所有格（'s）を落とす。残るアポストロフィでは Tokenize と同じく分ける。
func expandApostrophe(w string) []string {
	if e, ok := contractions[w]; ok {
		return strings.Fields(e)
	}
	if stem, ok := strings.CutSuffix(w, "n't"); ok && stem != "" {
		return []string{stem, "not"}
	}
	for _, p := range elisions {
		if rest, ok := strings.CutPrefix(w, p); ok && rest != "" {
			w = rest
			break
		}
	}
	w = strings.TrimSuffix(w, "'s")
	return strings.FieldsFunc(w, func(r rune) bool { return r == '\'' })
}

// matchSymbol は記号で切ると意味が壊れる塊（IPv4 アドレス / 略語 / ドメイン名）を先頭から探し、索引語と読んだバイト数を返す。
func matchSymbol(s string) (string, int) {
	if !hasInnerDot(s) {
		return "", 0
	}
	if loc := ipv4Re.FindStringIndex(s); loc != nil && endsChunk(s[loc[1]:]) && validIPv4(s[:loc[1]]) {
		return s[:loc[1]], loc[1]
	}
	if loc := acronymRe.FindStringIndex(s); loc != nil && endsChunk(s[loc[1]:]) {
		return strings.ToLower(strings.ReplaceAll(s[:loc[1]], ".", "")), loc[1]
	}
	if loc := domainRe.FindStringIndex(s); loc != nil && endsChunk(s[loc[1]:]) {
		return strings.ToLower(s[:loc[1]]), loc[1]
	}
	return "", 0
}

func hasInnerDot(s string) bool {
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch {
		case c == '.':
			return i+1 < len(s) && isASCIIAlnum(s[i+1])
		case c == '-' || isASCIIAlnum(c):
		default:
			return false
		}
	}
	return false
}

// endsChunk は塊の直後で語が続かないこと。1.2.3.4.5 の先頭 4 つや google.com.au の途中で切らないため。
func endsChunk(rest string) bool {
	r, size := utf8.DecodeRuneInString(rest)
	if unicode.IsLetter(r) || unicode.IsDigit(r) {
		return false
	}
	if r == '.' || r == '-' {
		return size >= len(rest) || !isASCIIAlnum(rest[size])
	}
	return true
}

// continuesChunk は直前が語に続くピリオドかハイフンのとき true。1.2.3.4.5 の 2 から塊を探し直さないため。
func continuesChunk(before string) bool {
	n := len(before)
	return n >= 2 && (before[n-1] == '.' || before[n-1] == '-') && isASCIIAlnum(before[n-2])
}

func validIPv4(s string) bool {
	for p := range strings.SplitSeq(s, ".") {
		if n, err := strconv.Atoi(p); err != nil || n > 255 {
			return false
		}
	}
	return true
}

func isASCIIAlnum(c byte) bool {
	return c >= '0' && c <= '9' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z'
}

func isASCIILowerWord(s string) bool {
	if s == "" {
		return false
	}
	for i := 0; i < len(s); i++ {
		if s[i] < 'a' || s[i] > 'z' {
			return false
		}
	}
	return true
}
