// Package dedup は、取り込んだ文書の完全重複（ハッシュ）と準重複（断片の集合、MinHash、LSH）を見つける部品と、既読の除外に使うブルームフィルタを持つ。
package dedup

import (
	"bytes"
	"crypto/sha256"
	"strings"

	"golang.org/x/net/html"

	"github.com/hiro8ma/agent/go/internal/crawl"
)

// Text は HTML のタグを外して本文の文字だけを取り出し、空白の並びを1つの空白にまとめる。script と style の中身は捨てる。
func Text(page []byte) string {
	z := html.NewTokenizer(bytes.NewReader(page))
	var b strings.Builder
	skip := 0
	for {
		switch z.Next() {
		case html.ErrorToken:
			return strings.Join(strings.Fields(b.String()), " ")
		case html.StartTagToken:
			if name, _ := z.TagName(); isRawText(name) {
				skip++
			}
		case html.EndTagToken:
			if name, _ := z.TagName(); isRawText(name) && skip > 0 {
				skip--
			}
		case html.TextToken:
			if skip == 0 {
				b.Write(z.Text())
				b.WriteByte(' ')
			}
		}
	}
}

func isRawText(tag []byte) bool {
	return string(tag) == "script" || string(tag) == "style"
}

// Fingerprint は Text で正規化した本文の SHA-256。タグと空白の違いだけの文書は同じ値になる。
func Fingerprint(page []byte) [sha256.Size]byte {
	return sha256.Sum256([]byte(Text(page)))
}

// Kind は Exact.Add が文書をどう判定したか。
type Kind int

const (
	New Kind = iota
	SameURL
	SameContent
)

func (k Kind) String() string {
	return [...]string{"新規", "URL の重複", "完全重複"}[k]
}

// Exact は URL の正規形と本文のハッシュで、取り込んだ文書の完全重複を見つける。
type Exact struct {
	norm   crawl.Normalizer
	urls   map[string]bool
	hashes map[[sha256.Size]byte]string
}

func NewExact(norm crawl.Normalizer) *Exact {
	return &Exact{norm: norm, urls: map[string]bool{}, hashes: map[[sha256.Size]byte]string{}}
}

// Add は文書を加え、判定と、同じ内容を先に持っていた URL（新規なら自分の正規形）を返す。
func (e *Exact) Add(rawURL string, page []byte) (Kind, string, error) {
	u, err := e.norm.Normalize(rawURL)
	if err != nil {
		return New, "", err
	}
	if e.urls[u] {
		return SameURL, u, nil
	}
	e.urls[u] = true
	h := Fingerprint(page)
	if first, ok := e.hashes[h]; ok {
		return SameContent, first, nil
	}
	e.hashes[h] = u
	return New, u, nil
}
