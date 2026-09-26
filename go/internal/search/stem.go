package search

import (
	"slices"
	"strings"
)

// PorterStem は Porter（1980）の語幹化を、Porter 自身の ANSI C 版と同じ 3 点の変更（bli → ble、logi → log、2 文字以下は変えない）込みでかける。
// 入力は英小文字だけの語を想定する。
func PorterStem(word string) string {
	if len(word) <= 2 {
		return word
	}
	p := porter{b: []byte(word), k: len(word) - 1}
	p.step1ab()
	if p.k > 0 {
		p.step1c()
		p.step2()
		p.step3()
		p.step4()
		p.step5()
	}
	return string(p.b[:p.k+1])
}

// porter の b[0:k+1] が今の語、j は ends が一致させた接尾辞の直前の位置。
type porter struct {
	b    []byte
	k, j int
}

func (p *porter) cons(i int) bool {
	switch p.b[i] {
	case 'a', 'e', 'i', 'o', 'u':
		return false
	case 'y':
		return i == 0 || !p.cons(i-1)
	}
	return true
}

// m は b[0:j+1] を [C](VC)^m[V] と見たときの m。
func (p *porter) m() int {
	n, i := 0, 0
	for ; i <= p.j && p.cons(i); i++ {
	}
	for {
		for ; i <= p.j && !p.cons(i); i++ {
		}
		if i > p.j {
			return n
		}
		n++
		for ; i <= p.j && p.cons(i); i++ {
		}
		if i > p.j {
			return n
		}
	}
}

func (p *porter) vowelInStem() bool {
	for i := 0; i <= p.j; i++ {
		if !p.cons(i) {
			return true
		}
	}
	return false
}

func (p *porter) doubleC(i int) bool {
	return i >= 1 && p.b[i] == p.b[i-1] && p.cons(i)
}

func (p *porter) cvc(i int) bool {
	if i < 2 || !p.cons(i) || p.cons(i-1) || !p.cons(i-2) {
		return false
	}
	c := p.b[i]
	return c != 'w' && c != 'x' && c != 'y'
}

func (p *porter) ends(s string) bool {
	n := len(s)
	if n > p.k+1 || string(p.b[p.k-n+1:p.k+1]) != s {
		return false
	}
	p.j = p.k - n
	return true
}

func (p *porter) setTo(s string) {
	p.b = append(p.b[:p.j+1], s...)
	p.k = p.j + len(s)
}

func (p *porter) replace(s string) {
	if p.m() > 0 {
		p.setTo(s)
	}
}

func (p *porter) step1ab() {
	if p.b[p.k] == 's' {
		switch {
		case p.ends("sses"):
			p.k -= 2
		case p.ends("ies"):
			p.setTo("i")
		case p.b[p.k-1] != 's':
			p.k--
		}
	}
	if p.ends("eed") {
		if p.m() > 0 {
			p.k--
		}
		return
	}
	if (p.ends("ed") || p.ends("ing")) && p.vowelInStem() {
		p.k = p.j
		switch {
		case p.ends("at"):
			p.setTo("ate")
		case p.ends("bl"):
			p.setTo("ble")
		case p.ends("iz"):
			p.setTo("ize")
		case p.doubleC(p.k):
			if c := p.b[p.k-1]; c != 'l' && c != 's' && c != 'z' {
				p.k--
			}
		case p.m() == 1 && p.cvc(p.k):
			p.setTo("e")
		}
	}
}

func (p *porter) step1c() {
	if p.ends("y") && p.vowelInStem() {
		p.b[p.k] = 'i'
	}
}

// replaceFirst は接尾辞の対を順に見て、最初に一致したものだけを置き換える。条件 m > 0 を満たさなくても、その先の対は見ない。
func (p *porter) replaceFirst(pairs ...string) {
	for i := 0; i+1 < len(pairs); i += 2 {
		if p.ends(pairs[i]) {
			p.replace(pairs[i+1])
			return
		}
	}
}

func (p *porter) step2() {
	switch p.b[p.k-1] {
	case 'a':
		p.replaceFirst("ational", "ate", "tional", "tion")
	case 'c':
		p.replaceFirst("enci", "ence", "anci", "ance")
	case 'e':
		p.replaceFirst("izer", "ize")
	case 'l':
		p.replaceFirst("bli", "ble", "alli", "al", "entli", "ent", "eli", "e", "ousli", "ous")
	case 'o':
		p.replaceFirst("ization", "ize", "ation", "ate", "ator", "ate")
	case 's':
		p.replaceFirst("alism", "al", "iveness", "ive", "fulness", "ful", "ousness", "ous")
	case 't':
		p.replaceFirst("aliti", "al", "iviti", "ive", "biliti", "ble")
	case 'g':
		p.replaceFirst("logi", "log")
	}
}

func (p *porter) step3() {
	switch p.b[p.k] {
	case 'e':
		p.replaceFirst("icate", "ic", "ative", "", "alize", "al")
	case 'i':
		p.replaceFirst("iciti", "ic")
	case 'l':
		p.replaceFirst("ical", "ic", "ful", "")
	case 's':
		p.replaceFirst("ness", "")
	}
}

var step4Suffixes = map[byte][]string{
	'a': {"al"},
	'c': {"ance", "ence"},
	'e': {"er"},
	'i': {"ic"},
	'l': {"able", "ible"},
	'n': {"ant", "ement", "ment", "ent"},
	's': {"ism"},
	't': {"ate", "iti"},
	'u': {"ous"},
	'v': {"ive"},
	'z': {"ize"},
}

func (p *porter) step4() {
	matched := false
	if p.b[p.k-1] == 'o' {
		matched = p.ends("ion") && p.j >= 0 && (p.b[p.j] == 's' || p.b[p.j] == 't') || p.ends("ou")
	} else {
		matched = slices.ContainsFunc(step4Suffixes[p.b[p.k-1]], p.ends)
	}
	if matched && p.m() > 1 {
		p.k = p.j
	}
}

func (p *porter) step5() {
	p.j = p.k
	if p.b[p.k] == 'e' {
		if a := p.m(); a > 1 || a == 1 && !p.cvc(p.k-1) {
			p.k--
		}
	}
	if p.b[p.k] == 'l' && p.doubleC(p.k) && p.m() > 1 {
		p.k--
	}
}

// irregularForms は規則で戻せない語形と基本形の対。品詞は見ないので saw は常に see にする。
var irregularForms = map[string]string{
	"am": "be", "is": "be", "are": "be", "was": "be", "were": "be", "been": "be", "being": "be",
	"has": "have", "had": "have", "having": "have",
	"does": "do", "did": "do", "done": "do",
	"goes": "go", "went": "go", "gone": "go",
	"saw": "see", "seen": "see",
	"ate": "eat", "eaten": "eat",
	"took": "take", "taken": "take",
	"gave": "give", "given": "give",
	"came": "come",
	"ran":  "run", "running": "run",
	"made": "make", "making": "make",
	"wrote": "write", "written": "write", "writing": "write",
	"bought": "buy", "thought": "think", "taught": "teach",
	"children": "child", "men": "man", "women": "woman", "people": "person",
	"mice": "mouse", "feet": "foot", "teeth": "tooth", "geese": "goose",
	"better": "good", "best": "good",
}

// Lemmatize は不規則変化の辞書を先に引き、無ければ -s / -es / -ies / -ed / -ing を規則で戻す。入力は英小文字だけの語を想定する。
func Lemmatize(word string) string {
	if base, ok := irregularForms[word]; ok {
		return base
	}
	switch {
	case len(word) > 4 && strings.HasSuffix(word, "ies"):
		return word[:len(word)-3] + "y"
	case len(word) > 4 && hasSibilantES(word):
		return word[:len(word)-2]
	case len(word) > 3 && strings.HasSuffix(word, "s") && !hasAnySuffix(word, "ss", "us", "is"):
		return word[:len(word)-1]
	case len(word) > 4 && strings.HasSuffix(word, "ied"):
		return word[:len(word)-3] + "y"
	case len(word) > 4 && strings.HasSuffix(word, "ed") && !strings.HasSuffix(word, "eed"):
		return restoreStem(word, word[:len(word)-2])
	case len(word) > 5 && strings.HasSuffix(word, "ing"):
		return restoreStem(word, word[:len(word)-3])
	}
	return word
}

func hasSibilantES(w string) bool {
	return hasAnySuffix(w, "sses", "xes", "zes", "ches", "shes")
}

func hasAnySuffix(w string, suffixes ...string) bool {
	for _, s := range suffixes {
		if strings.HasSuffix(w, s) {
			return true
		}
	}
	return false
}

// restoreStem は -ed / -ing を外した残りを基本形に寄せる。stopp → stop、hop → hope、creat → create。
func restoreStem(word, s string) string {
	if !strings.ContainsAny(s, "aeiouy") {
		return word
	}
	n := len(s)
	switch {
	case hasAnySuffix(s, "at", "bl", "iz", "v"):
		return s + "e"
	case n >= 2 && s[n-1] == s[n-2] && !strings.ContainsRune("aeioulsz", rune(s[n-1])):
		return s[:n-1]
	case n == 3 && isConsonant(s[0]) && !isConsonant(s[1]) && isConsonant(s[2]) && !strings.ContainsRune("wxy", rune(s[2])):
		return s + "e"
	}
	return s
}

func isConsonant(c byte) bool {
	return !strings.ContainsRune("aeiou", rune(c))
}
