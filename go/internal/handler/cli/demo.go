// Package cli は 3 実装共通の CLI ハンドラ。
package cli

import (
	"context"
	"fmt"
	"log"

	"github.com/hiro8ma/agent/go/internal/agent/domain/model"
	"github.com/hiro8ma/agent/go/internal/agent/usecase"
)

// Demo は事前定義クエリを順次流すデモランナー。
type Demo struct {
	label string
	ask   usecase.Ask
}

// NewDemo は Demo ランナーを生成する。label は実装名の表示ラベル（genai / genkit / adk）。
func NewDemo(label string, ask usecase.Ask) *Demo {
	return &Demo{label: label, ask: ask}
}

// Run はユーザー質問のシーケンスをマルチターンで処理する。
func (d *Demo) Run(ctx context.Context) {
	fmt.Printf("=== Demo [%s] ===\n", d.label)

	queries := []string{
		"123 と 456 を足して、その結果を 2 倍してください。計算には calculator を使ってください。",
		"Genkit とは何ですか？search_knowledge で調べてください。",
	}

	conv := &model.Conversation{}
	for _, q := range queries {
		fmt.Println("\n--- Q", q)
		out, err := d.ask.Handle(ctx, usecase.AskInput{
			UserMessage:  q,
			Conversation: conv,
		})
		if err != nil {
			log.Printf("error %v", err)
			continue
		}
		conv = out.Conversation
		fmt.Printf("--- A (iter=%d, tokens=%d)\n", out.Iterations, out.TotalTokens)
		fmt.Println(out.Answer)
	}
}
