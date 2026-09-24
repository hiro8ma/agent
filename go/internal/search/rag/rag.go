// Package rag は検索で取った文書だけを根拠に Genkit のモデルで答え、根拠にした文書の ID を構造化出力で返させる最小の RAG。
package rag

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"

	"github.com/hiro8ma/agent/go/internal/search"
)

const DefaultLimit = 3

// NoContextAnswer は検索で文書が 1 件も取れなかったときの回答。根拠の無いまま生成させるとモデルの知識で答えてしまうため、モデルを呼ばない。
const NoContextAnswer = "資料に該当する記述が見つからなかった"

const system = "あなたは与えられた資料だけを根拠に質問に答える担当です。" +
	"資料に書かれていないことは推測で補わず、資料からは分からないと答えてください。" +
	"source_ids には回答の根拠にした資料の ID だけを入れてください。"

// Output はモデルに返させる構造化出力。
type Output struct {
	Answer    string   `json:"answer" jsonschema_description:"資料だけを根拠にした回答"`
	SourceIDs []string `json:"source_ids" jsonschema_description:"回答の根拠にした資料の ID。資料の見出しの [] の中の値"`
}

// Answer の Sources は渡した文書のうちモデルが根拠に挙げたもの。Rejected は渡していない ID で、モデルの作り話として取り除いた。
type Answer struct {
	Text      string
	Sources   []search.Doc
	Retrieved []search.Doc
	Rejected  []string
	Usage     *ai.GenerationUsage
}

// RAG の Limit が 0 以下なら DefaultLimit 件を取る。Options はモデルの指定など生成に足す設定。
type RAG struct {
	G         *genkit.Genkit
	Retriever search.Retriever
	Limit     int
	Options   []ai.GenerateOption
}

func (r RAG) Ask(ctx context.Context, question string) (Answer, error) {
	if r.G == nil || r.Retriever == nil {
		return Answer{}, errors.New("rag: G and Retriever are required")
	}
	limit := r.Limit
	if limit <= 0 {
		limit = DefaultLimit
	}
	docs, err := r.Retriever.Search(ctx, question, limit)
	if err != nil {
		return Answer{}, fmt.Errorf("rag: retrieve: %w", err)
	}
	if len(docs) == 0 {
		return Answer{Text: NoContextAnswer}, nil
	}
	out, resp, err := genkit.GenerateData[Output](ctx, r.G, append([]ai.GenerateOption{
		ai.WithSystem(system),
		ai.WithPrompt(Prompt(question, docs)),
	}, r.Options...)...)
	if err != nil {
		return Answer{}, fmt.Errorf("rag: generate: %w", err)
	}
	a := Answer{Text: out.Answer, Retrieved: docs, Usage: resp.Usage}
	a.Sources, a.Rejected = ground(out.SourceIDs, docs)
	return a, nil
}

// Prompt は文書を ID つきの見出しで並べ、最後に質問を置く。ID の無い文書は並び順の番号で呼ぶ。
func Prompt(question string, docs []search.Doc) string {
	var b strings.Builder
	b.WriteString("# 資料\n")
	for i, d := range docs {
		fmt.Fprintf(&b, "\n## [%s]", docID(d, i))
		if d.Title != "" {
			fmt.Fprintf(&b, " %s", d.Title)
		}
		fmt.Fprintf(&b, "\n%s\n", d.Content)
	}
	fmt.Fprintf(&b, "\n# 質問\n%s\n", question)
	return b.String()
}

func docID(d search.Doc, i int) string {
	if d.ID != "" {
		return d.ID
	}
	return fmt.Sprintf("doc%d", i+1)
}

// ground はモデルの挙げた ID を渡した文書に引き当てる。渡していない ID は捨て、同じ ID は 1 回だけ数える。
func ground(ids []string, docs []search.Doc) ([]search.Doc, []string) {
	byID := make(map[string]search.Doc, len(docs))
	for i, d := range docs {
		byID[docID(d, i)] = d
	}
	var (
		sources  []search.Doc
		rejected []string
		seen     []string
	)
	for _, id := range ids {
		id = strings.Trim(strings.TrimSpace(id), "[]")
		if slices.Contains(seen, id) {
			continue
		}
		seen = append(seen, id)
		if d, ok := byID[id]; ok {
			sources = append(sources, d)
		} else {
			rejected = append(rejected, id)
		}
	}
	return sources, rejected
}
