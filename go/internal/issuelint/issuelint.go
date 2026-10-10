// Package issuelint は、エージェントに渡す前の Issue の本文が、依頼ではなく定義になっているかを確かめる。
// 必須の見出し（Goal / Context Pointers / Constraints / Done When）とその本文、成果物がコード変更を含まないことの明記を求め、エージェントの範囲を広げようとする語を拾う。
//
//	## Goal
//	検索結果の並び順を説明する文書を作る
//	## Context Pointers
//	internal/search/
//	## Constraints
//	コード変更は行わない
//	## Done When
//	docs/ に提案の文書が 1 本ある
package issuelint

import (
	"fmt"
	"regexp"
	"strings"
)

// NoCodeChangeNotice は、成果物がコード変更を含まないと書いた行が無いときに Missing に入る名前。
const NoCodeChangeNotice = "成果物がコード変更を含まないことの明記"

// NoCodeChangePhrases は、成果物がコード変更を含まないと書いた行とみなす語。
var NoCodeChangePhrases = []string{
	"コード変更は行わない",
	"コードは変更しない",
	"no code changes",
	"実装変更は行わない",
}

// ForbiddenPhrases は、読み取りだけの提案の作業でエージェントに従わせない語。呼び出し側で差し替えられる。
var ForbiddenPhrases = []string{
	"ignore previous",
	"ignore the above",
	"以前の指示を無視",
	"上の指示を無視",
	"git push",
	"force push",
	"rm -rf",
	"curl ",
	"wget ",
	"secrets.",
	"GITHUB_TOKEN",
}

type section struct {
	name    string
	aliases []string
}

var sections = []section{
	{name: "Goal", aliases: []string{"目的", "ゴール"}},
	{name: "Context Pointers", aliases: []string{"参照先", "対象", "Context"}},
	{name: "Constraints", aliases: []string{"守る条件", "制約", "注意事項"}},
	{name: "Done When", aliases: []string{"完了条件", "Done"}},
}

var (
	headingRe  = regexp.MustCompile(`^#{1,6}(\s|$)`)
	requiredRe = regexp.MustCompile(`^#{2,3}\s+(.*)$`)
	commentRe  = regexp.MustCompile(`<!--.*?-->`)
)

// Hit は禁止の語を含む 1 行。Line は 1 から数える。
type Hit struct {
	Line int
	Text string
}

// Result は Check の結果。
type Result struct {
	Missing   []string
	Empty     []string
	Forbidden []Hit
}

// OK は問題が 1 件も無いときに true を返す。
func (r Result) OK() bool {
	return len(r.Missing) == 0 && len(r.Empty) == 0 && len(r.Forbidden) == 0
}

func (r Result) String() string {
	var b strings.Builder
	if len(r.Missing) > 0 {
		fmt.Fprintf(&b, "不足: %s\n", strings.Join(r.Missing, ", "))
	}
	if len(r.Empty) > 0 {
		fmt.Fprintf(&b, "空: %s\n", strings.Join(r.Empty, ", "))
	}
	for _, h := range r.Forbidden {
		fmt.Fprintf(&b, "禁止の語 %d行目: %s\n", h.Line, h.Text)
	}
	return b.String()
}

// Check は Issue の本文を確かめる。
func Check(body string) Result {
	lines := strings.Split(strings.ReplaceAll(body, "\r\n", "\n"), "\n")

	found := map[string]bool{}
	filled := map[string]bool{}
	current := ""
	inComment := false
	noticed := false
	var r Result
	for i, line := range lines {
		lower := strings.ToLower(line)
		for _, p := range ForbiddenPhrases {
			if strings.Contains(lower, strings.ToLower(p)) {
				r.Forbidden = append(r.Forbidden, Hit{Line: i + 1, Text: strings.TrimSpace(line)})
				break
			}
		}
		for _, p := range NoCodeChangePhrases {
			if strings.Contains(lower, strings.ToLower(p)) {
				noticed = true
			}
		}

		if headingRe.MatchString(line) {
			current = ""
			if m := requiredRe.FindStringSubmatch(line); m != nil {
				current = sectionName(m[1])
				if current != "" {
					found[current] = true
				}
			}
			continue
		}
		var text string
		text, inComment = stripComment(line, inComment)
		if current != "" && strings.TrimSpace(text) != "" {
			filled[current] = true
		}
	}

	for _, s := range sections {
		switch {
		case !found[s.name]:
			r.Missing = append(r.Missing, s.name)
		case !filled[s.name]:
			r.Empty = append(r.Empty, s.name)
		}
	}
	if !noticed {
		r.Missing = append(r.Missing, NoCodeChangeNotice)
	}
	return r
}

func sectionName(heading string) string {
	h := strings.TrimRight(heading, " \t:：")
	for _, s := range sections {
		if strings.EqualFold(h, s.name) {
			return s.name
		}
		for _, a := range s.aliases {
			if strings.EqualFold(h, a) {
				return s.name
			}
		}
	}
	return ""
}

// stripComment は Issue テンプレートの HTML コメントを除く。コメントの書き方の案内だけでは本文を書いたことにしない。
func stripComment(line string, inComment bool) (string, bool) {
	if inComment {
		end := strings.Index(line, "-->")
		if end < 0 {
			return "", true
		}
		line = line[end+len("-->"):]
	}
	line = commentRe.ReplaceAllString(line, "")
	if start := strings.Index(line, "<!--"); start >= 0 {
		return line[:start], true
	}
	return line, false
}
