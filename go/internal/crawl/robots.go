// Package crawl は、先方への負荷を抑えるクローラーの部品（robots.txt の解釈、URL の正規化、ホストごとの待ち行列）と、取り直しの方針ごとの鮮度の計算を持つ。
package crawl

import (
	"slices"
	"strings"
)

// StatusUnreachable は、名前の解決や接続に失敗して応答の状態が無いことを表す。
const StatusUnreachable = 0

type robotsRule struct {
	allow   bool
	pattern string
}

type robotsGroup struct {
	agents []string
	rules  []robotsRule
}

// Robots は RFC 9309 の robots.txt の規則。
type Robots struct {
	groups      []robotsGroup
	disallowAll bool
}

// RobotsFromFetch は robots.txt の取得の状態と本文から規則を作る。
// 2xx は本文を解釈し、4xx は robots.txt が無いとみなして全部許可、5xx と到達できない場合は全部禁止にする。
func RobotsFromFetch(status int, body []byte) *Robots {
	switch {
	case status >= 200 && status < 300:
		return ParseRobots(string(body))
	case status >= 300 && status < 500:
		// 3xx が残るのはリダイレクトを辿りきれなかった場合で、RFC 9309 は無いものとみなしてよいとする。
		return &Robots{}
	default:
		return &Robots{disallowAll: true}
	}
}

// ParseRobots は robots.txt の本文を解釈する。User-agent の行が続く間は同じグループにまとめ、規則の行の後の User-agent から次のグループにする。
func ParseRobots(body string) *Robots {
	r := &Robots{}
	cur := -1
	inRules := false
	for line := range strings.Lines(strings.TrimPrefix(body, "\uFEFF")) {
		line, _, _ = strings.Cut(line, "#")
		key, value, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		value = strings.TrimSpace(value)
		switch strings.ToLower(strings.TrimSpace(key)) {
		case "user-agent":
			if cur < 0 || inRules {
				r.groups = append(r.groups, robotsGroup{})
				cur = len(r.groups) - 1
				inRules = false
			}
			r.groups[cur].agents = append(r.groups[cur].agents, productToken(value))
		case "allow", "disallow":
			if cur < 0 {
				continue
			}
			inRules = true
			if value == "" {
				continue
			}
			r.groups[cur].rules = append(r.groups[cur].rules, robotsRule{
				allow:   strings.EqualFold(strings.TrimSpace(key), "allow"),
				pattern: encodePath(value),
			})
		}
	}
	return r
}

// productToken は User-agent の値から、英字 / _ / - が続く先頭の部分を小文字で返す。
func productToken(value string) string {
	if strings.HasPrefix(value, "*") {
		return "*"
	}
	end := strings.IndexFunc(value, func(c rune) bool {
		return (c < 'a' || c > 'z') && (c < 'A' || c > 'Z') && c != '_' && c != '-'
	})
	if end >= 0 {
		value = value[:end]
	}
	return strings.ToLower(value)
}

// Allowed は agent（クローラーの名前）が path（パスとクエリ）を取ってよいかを返す。
// 最も長く一致した規則が勝ち、同じ長さなら Allow が勝つ。どの規則にも一致しなければ許可する。
func (r *Robots) Allowed(agent, path string) bool {
	if path == "/robots.txt" {
		return true
	}
	if r.disallowAll {
		return false
	}
	path = encodePath(path)
	best := -1
	allowed := true
	for _, rule := range r.rulesFor(agent) {
		n := len(rule.pattern)
		if n < best || !matchPattern(rule.pattern, path) {
			continue
		}
		if n > best || rule.allow {
			best = n
			allowed = rule.allow
		}
	}
	return allowed
}

// rulesFor は agent に合うグループの規則をすべて合わせて返す。合うグループが無ければ * のグループを使う。
func (r *Robots) rulesFor(agent string) []robotsRule {
	agent = strings.ToLower(agent)
	collect := func(name string) ([]robotsRule, bool) {
		var rules []robotsRule
		found := false
		for _, g := range r.groups {
			if slices.Contains(g.agents, name) {
				rules = append(rules, g.rules...)
				found = true
			}
		}
		return rules, found
	}
	if rules, ok := collect(agent); ok {
		return rules
	}
	rules, _ := collect("*")
	return rules
}

// matchPattern はパスの先頭から pattern を当てる。* は任意の文字列、末尾の $ はパスの終わりに一致する。
func matchPattern(pattern, path string) bool {
	if p, ok := strings.CutSuffix(pattern, "$"); ok {
		pattern = p
	} else {
		pattern += "*"
	}
	p, s := 0, 0
	star, mark := -1, 0
	for s < len(path) {
		switch {
		case p < len(pattern) && pattern[p] == '*':
			star, mark = p, s
			p++
		case p < len(pattern) && pattern[p] == path[s]:
			p++
			s++
		case star >= 0:
			p = star + 1
			mark++
			s = mark
		default:
			return false
		}
	}
	for p < len(pattern) && pattern[p] == '*' {
		p++
	}
	return p == len(pattern)
}

// encodePath は ASCII でない文字を %XX にし、既存の %xx の16進を大文字にそろえて、規則とパスを同じ表記で比べられるようにする。
func encodePath(s string) string {
	const hex = "0123456789ABCDEF"
	var b strings.Builder
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch {
		case c >= 0x80:
			b.WriteByte('%')
			b.WriteByte(hex[c>>4])
			b.WriteByte(hex[c&0xF])
		case c == '%' && i+2 < len(s) && isHex(s[i+1]) && isHex(s[i+2]):
			b.WriteByte('%')
			b.WriteString(strings.ToUpper(s[i+1 : i+3]))
			i += 2
		default:
			b.WriteByte(c)
		}
	}
	return b.String()
}

func isHex(c byte) bool {
	return ('0' <= c && c <= '9') || ('a' <= c && c <= 'f') || ('A' <= c && c <= 'F')
}
