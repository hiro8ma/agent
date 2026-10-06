// Package wikilint は、LLM Wiki のページが根拠にした出典の内容が変わっていないかを、ハッシュで確かめる。
// 各ページの先頭の frontmatter に、根拠にしたファイル（file）か別のページ（page）と、その sha256 を書く。
//
//	---
//	sources:
//	  - file: ../src/order.go
//	    sha256: 9f86d0...
//	  - page: billing/refund.md
//	    sha256: 2c26b4...
//	---
//
// ファイルは中身全体、ページは frontmatter を除いた本文のハッシュを使う。出典の欄を書き直しただけで、下流のページが古くならないようにするため。
package wikilint

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"
)

// InboxDir は確認前の回答を置く一時置き場。ここにあるものはページの根拠にできない。
const InboxDir = "inbox"

// Source はページが根拠にした 1 つの出典。File と Page のどちらか一方だけを持つ。
type Source struct {
	File   string
	Page   string
	SHA256 string
}

func (s Source) target() string {
	if s.File != "" {
		return "file " + s.File
	}
	return "page " + s.Page
}

// Page は wiki の 1 ページ。Path は wiki の根からの相対パス。
type Page struct {
	Path    string
	Sources []Source
	Body    string
	// Other は frontmatter のうち sources 以外の行。Stamp で書き直すときにそのまま残す。
	Other []string
}

// Problem は lint で見つかった 1 件の問題。
type Problem struct {
	Page   string
	Source string
	Kind   string // stale / missing / inbox / unhashed
	Detail string
}

func (p Problem) String() string {
	return fmt.Sprintf("%s: %s %s（%s）", p.Page, p.Kind, p.Source, p.Detail)
}

// Load は root の下の .md をすべて読む。inbox の下は読まない。
func Load(root string) (map[string]*Page, error) {
	pages := map[string]*Page{}
	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(root, path)
		if d.IsDir() {
			if rel == InboxDir {
				return filepath.SkipDir
			}
			return nil
		}
		if filepath.Ext(path) != ".md" {
			return nil
		}
		raw, err := os.ReadFile(path) //nolint:gosec // wiki の根は実行する人が指定する
		if err != nil {
			return err
		}
		p, err := Parse(filepath.ToSlash(rel), string(raw))
		if err != nil {
			return err
		}
		pages[p.Path] = p
		return nil
	})
	return pages, err
}

// Parse はページの frontmatter から sources を読み、本文と分ける。frontmatter が無ければ出典なしのページとして扱う。
func Parse(path, raw string) (*Page, error) {
	p := &Page{Path: path, Body: raw}
	if !strings.HasPrefix(raw, "---\n") {
		return p, nil
	}
	end := strings.Index(raw[4:], "\n---\n")
	if end < 0 {
		return nil, fmt.Errorf("%s: frontmatter が閉じていない", path)
	}
	front := raw[4 : 4+end]
	p.Body = raw[4+end+len("\n---\n"):]
	inSources := false
	sc := bufio.NewScanner(strings.NewReader(front))
	for sc.Scan() {
		line := sc.Text()
		trimmed := strings.TrimSpace(line)
		switch {
		case trimmed == "sources:":
			inSources = true
		case !strings.HasPrefix(line, " ") && trimmed != "":
			inSources = false
			p.Other = append(p.Other, line)
		case !inSources:
			p.Other = append(p.Other, line)
		case inSources && strings.HasPrefix(trimmed, "- "):
			key, val, ok := keyValue(strings.TrimPrefix(trimmed, "- "))
			if !ok {
				return nil, fmt.Errorf("%s: 出典の行を読めない: %q", path, line)
			}
			s := Source{}
			switch key {
			case "file":
				s.File = val
			case "page":
				s.Page = val
			default:
				return nil, fmt.Errorf("%s: 出典は file か page のどちらか: %q", path, line)
			}
			p.Sources = append(p.Sources, s)
		case inSources && trimmed != "":
			key, val, ok := keyValue(trimmed)
			if !ok || key != "sha256" || len(p.Sources) == 0 {
				return nil, fmt.Errorf("%s: 出典の行を読めない: %q", path, line)
			}
			p.Sources[len(p.Sources)-1].SHA256 = val
		}
	}
	return p, nil
}

func keyValue(s string) (string, string, bool) {
	key, val, ok := strings.Cut(s, ":")
	return strings.TrimSpace(key), strings.Trim(strings.TrimSpace(val), `"`), ok
}

// Hash は出典の今のハッシュを返す。ファイルは中身全体、ページは本文だけを使う。
func Hash(root string, pages map[string]*Page, page *Page, s Source) (string, error) {
	if s.Page != "" {
		up, ok := pages[s.Page]
		if !ok {
			return "", fs.ErrNotExist
		}
		return digest([]byte(up.Body)), nil
	}
	raw, err := os.ReadFile(filepath.Join(root, filepath.Dir(page.Path), s.File)) //nolint:gosec // 出典はページに書かれた根拠で、wiki の外のコードも読むのが目的
	if err != nil {
		return "", err
	}
	return digest(raw), nil
}

func digest(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// Lint は各ページの出典のハッシュを計算し直し、問題を返す。問題が無ければ空。
func Lint(root string, pages map[string]*Page) []Problem {
	var out []Problem
	for _, path := range sortedKeys(pages) {
		page := pages[path]
		for _, s := range page.Sources {
			if isInbox(page, s) {
				out = append(out, Problem{Page: path, Source: s.target(), Kind: "inbox", Detail: "一時置き場のものは根拠にできない"})
				continue
			}
			got, err := Hash(root, pages, page, s)
			switch {
			case errors.Is(err, fs.ErrNotExist):
				out = append(out, Problem{Page: path, Source: s.target(), Kind: "missing", Detail: "出典が見つからない"})
			case err != nil:
				out = append(out, Problem{Page: path, Source: s.target(), Kind: "missing", Detail: err.Error()})
			case s.SHA256 == "":
				out = append(out, Problem{Page: path, Source: s.target(), Kind: "unhashed", Detail: "ハッシュが書かれていない"})
			case got != s.SHA256:
				out = append(out, Problem{Page: path, Source: s.target(), Kind: "stale", Detail: "出典が変わったので読み直す"})
			}
		}
	}
	return out
}

func isInbox(page *Page, s Source) bool {
	target := s.Page
	if s.File != "" {
		target = filepath.ToSlash(filepath.Clean(filepath.Join(filepath.Dir(page.Path), s.File)))
	}
	return target == InboxDir || strings.HasPrefix(target, InboxDir+"/")
}

// Downstream は、path を直接または間接に根拠にしているページを、近い順に返す。path を直すと、これらが順に古くなる。
func Downstream(pages map[string]*Page, path string) []string {
	seen := map[string]bool{path: true}
	var out []string
	queue := []string{path}
	for len(queue) > 0 {
		cur := queue[0]
		queue = queue[1:]
		for _, p := range sortedKeys(pages) {
			if seen[p] {
				continue
			}
			if slices.ContainsFunc(pages[p].Sources, func(s Source) bool { return s.Page == cur }) {
				seen[p] = true
				out = append(out, p)
				queue = append(queue, p)
			}
		}
	}
	return out
}

// Stamp は、読み直して直したページの出典のハッシュを今の値に書き換えたページの全文を返す。
func Stamp(root string, pages map[string]*Page, path string) (string, error) {
	page, ok := pages[path]
	if !ok {
		return "", fmt.Errorf("%s: ページが無い", path)
	}
	var b strings.Builder
	b.WriteString("---\n")
	for _, line := range page.Other {
		b.WriteString(line + "\n")
	}
	b.WriteString("sources:\n")
	for _, s := range page.Sources {
		if isInbox(page, s) {
			return "", fmt.Errorf("%s: 一時置き場のもの（%s）は根拠にできない", path, s.target())
		}
		got, err := Hash(root, pages, page, s)
		if err != nil {
			return "", fmt.Errorf("%s: %s: %w", path, s.target(), err)
		}
		if s.File != "" {
			fmt.Fprintf(&b, "  - file: %s\n", s.File)
		} else {
			fmt.Fprintf(&b, "  - page: %s\n", s.Page)
		}
		fmt.Fprintf(&b, "    sha256: %s\n", got)
	}
	b.WriteString("---\n")
	b.WriteString(page.Body)
	return b.String(), nil
}

func sortedKeys(pages map[string]*Page) []string {
	keys := make([]string, 0, len(pages))
	for k := range pages {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	return keys
}
