// gen は e-Gov 法令 API v2 から日本国憲法を取得し、条ごとに 1 つの文字列にした JSON の配列を書き出す。
// 出典は https://laws.e-gov.go.jp/api/2/law_data/321CONSTITUTION?law_full_text_format=json（2026-10-03 取得）。
// 法令は著作権法第 13 条により著作権の目的とならないため、生成物をリポジトリに置く。
//
//	go generate ./internal/enginecmp
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"
)

const sourceURL = "https://laws.e-gov.go.jp/api/2/law_data/321CONSTITUTION?law_full_text_format=json"

// node は法令 XML を JSON にした木の 1 要素。children には文字列と node が混ざる。
type node struct {
	Tag      string            `json:"tag"`
	Attr     map[string]string `json:"attr"`
	Children []json.RawMessage `json:"children"`
}

func main() {
	out := flag.String("out", "kenpou.json", "書き出す先")
	flag.Parse()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	body, err := fetch(ctx)
	if err != nil {
		log.Fatal(err)
	}
	articles, err := extract(body)
	if err != nil {
		log.Fatal(err)
	}
	b, err := json.MarshalIndent(articles, "", "  ")
	if err != nil {
		log.Fatal(err)
	}
	if err := os.WriteFile(*out, append(b, '\n'), 0o600); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%d 条を %s に書き出した\n", len(articles), *out)
}

func fetch(ctx context.Context) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, sourceURL, nil)
	if err != nil {
		return nil, err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("e-Gov 法令 API: %s", resp.Status)
	}
	return io.ReadAll(resp.Body)
}

func extract(body []byte) ([]string, error) {
	var res struct {
		LawFullText node `json:"law_full_text"`
	}
	if err := json.Unmarshal(body, &res); err != nil {
		return nil, err
	}
	var articles []string
	var walk func(n node) error
	walk = func(n node) error {
		if n.Tag == "Article" {
			if want := strconv.Itoa(len(articles) + 1); n.Attr["Num"] != want {
				return fmt.Errorf("条の番号が %s のはずが %q", want, n.Attr["Num"])
			}
			articles = append(articles, article(n))
			return nil
		}
		for _, c := range children(n) {
			if err := walk(c); err != nil {
				return err
			}
		}
		return nil
	}
	if err := walk(res.LawFullText); err != nil {
		return nil, err
	}
	if len(articles) != 103 {
		return nil, fmt.Errorf("条の数が 103 のはずが %d", len(articles))
	}
	return articles, nil
}

// article は「第一条　本文」の形にし、項と号は改行で区切る。憲法の項には番号の文字が無いので項番号は付けない。
func article(n node) string {
	var title string
	var lines []string
	for _, c := range children(n) {
		switch c.Tag {
		case "ArticleTitle":
			title = text(c)
		case "Paragraph":
			for _, p := range children(c) {
				switch p.Tag {
				case "ParagraphSentence":
					lines = append(lines, text(p))
				case "Item":
					lines = append(lines, item(p))
				}
			}
		}
	}
	return title + "　" + strings.Join(lines, "\n")
}

func item(n node) string {
	var title, sentence string
	for _, c := range children(n) {
		switch c.Tag {
		case "ItemTitle":
			title = text(c)
		case "ItemSentence":
			sentence = text(c)
		}
	}
	return title + "　" + sentence
}

// text はふりがな（Rt）を落として文字列をつなぐ。
func text(n node) string {
	var b strings.Builder
	for _, raw := range n.Children {
		var s string
		if json.Unmarshal(raw, &s) == nil {
			b.WriteString(s)
			continue
		}
		var c node
		if json.Unmarshal(raw, &c) == nil && c.Tag != "Rt" {
			b.WriteString(text(c))
		}
	}
	return b.String()
}

func children(n node) []node {
	var out []node
	for _, raw := range n.Children {
		var c node
		if json.Unmarshal(raw, &c) == nil && c.Tag != "" {
			out = append(out, c)
		}
	}
	return out
}
