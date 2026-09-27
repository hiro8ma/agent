package morph

import (
	"unicode"
	"unicode/utf8"
)

// CharType は文字種。CharMixed は複数の文字種を含むトークン。
type CharType int

const (
	CharKanji CharType = iota
	CharHiragana
	CharKatakana
	CharAlpha
	CharDigit
	CharOther
	CharMixed
	numCharTypes
)

const prolongedSoundMark = 'ー'

func runeType(r rune) CharType {
	switch {
	case unicode.Is(unicode.Han, r) || r == '々':
		return CharKanji
	case unicode.Is(unicode.Hiragana, r):
		return CharHiragana
	case unicode.Is(unicode.Katakana, r) || r == prolongedSoundMark:
		return CharKatakana
	case unicode.IsDigit(r):
		return CharDigit
	case unicode.Is(unicode.Latin, r):
		return CharAlpha
	default:
		return CharOther
	}
}

// runeTypes は長音符 ー を直前の文字と同じ文字種にする。ひらがなの「らーめん」もカタカナとの混在にしないため。
func runeTypes(rs []rune) []CharType {
	ts := make([]CharType, len(rs))
	for i, r := range rs {
		ts[i] = runeType(r)
		if r == prolongedSoundMark && i > 0 {
			ts[i] = ts[i-1]
		}
	}
	return ts
}

// CharTypeOf は s の文字種を返す。すべての文字が同じ文字種ならその文字種、そうでなければ CharMixed。
func CharTypeOf(s string) CharType {
	ts := runeTypes([]rune(s))
	if len(ts) == 0 {
		return CharOther
	}
	for _, t := range ts[1:] {
		if t != ts[0] {
			return CharMixed
		}
	}
	return ts[0]
}

const numLengthBuckets = 4

// lengthBucket は文字数を 1 / 2 / 3 / 4 以上の 4 つに分ける。
func lengthBucket(s string) int {
	return min(max(utf8.RuneCountInString(s), 1), numLengthBuckets) - 1
}
