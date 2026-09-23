package search

import (
	"strings"
	"unicode"
)

// Tokenize は英数字を小文字の単語に、日本語（ひらがな / カタカナ / 漢字）を文字の bigram に分ける。
func Tokenize(text string) []string {
	var (
		tokens []string
		word   strings.Builder
		ja     []rune
	)
	flushWord := func() {
		if word.Len() > 0 {
			tokens = append(tokens, word.String())
			word.Reset()
		}
	}
	flushJa := func() {
		switch len(ja) {
		case 0:
		case 1:
			tokens = append(tokens, string(ja))
		default:
			for i := range len(ja) - 1 {
				tokens = append(tokens, string(ja[i:i+2]))
			}
		}
		ja = ja[:0]
	}
	for _, r := range text {
		switch {
		case isJapanese(r):
			flushWord()
			ja = append(ja, r)
		case unicode.IsLetter(r) || unicode.IsDigit(r):
			flushJa()
			word.WriteRune(unicode.ToLower(r))
		default:
			flushWord()
			flushJa()
		}
	}
	flushWord()
	flushJa()
	return tokens
}

func isJapanese(r rune) bool {
	return r == 'ー' || unicode.In(r, unicode.Hiragana, unicode.Katakana, unicode.Han)
}
