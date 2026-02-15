package genkit

import (
	"context"
	"fmt"
	"log"

	"github.com/firebase/genkit/go/ai"
	genkitsdk "github.com/firebase/genkit/go/genkit"
)

// RunDemo はデモモードを実行
func (a *App) RunDemo(ctx context.Context) {
	// === デモ1: シンプルなテキスト生成 ===
	fmt.Println("\n=== デモ1: シンプルなテキスト生成 ===")
	response, err := genkitsdk.GenerateText(ctx, a.g,
		ai.WithModel(a.model),
		ai.WithPrompt("Goプログラミング言語の良いところを3つ、簡潔に教えてください。"),
	)
	if err != nil {
		log.Printf("Failed to generate text: %v", err)
	} else {
		fmt.Println(response)
	}

	// === デモ2: Tool Calling（計算エージェント）===
	fmt.Println("\n=== デモ2: Tool Calling（計算エージェント）===")
	calcResponse, err := genkitsdk.Generate(ctx, a.g,
		ai.WithModel(a.model),
		ai.WithPrompt("123と456を足して、その結果を2倍してください。計算にはcalculatorツールを使ってください。"),
		ai.WithTools(a.calcTool),
	)
	if err != nil {
		log.Printf("Failed to generate with tools: %v", err)
	} else {
		fmt.Println("回答:", calcResponse.Text())
	}

	// === デモ3: 構造化出力（感情分析）===
	fmt.Println("\n=== デモ3: 構造化出力（感情分析）===")
	sentimentResult, _, err := genkitsdk.GenerateData[SentimentOutput](ctx, a.g,
		ai.WithModel(a.model),
		ai.WithPrompt(`
以下のテキストの感情を分析してください：
"今日はとても素晴らしい天気で、気分が良いです！"

JSON形式で回答してください。
`),
	)
	if err != nil {
		log.Printf("Failed sentiment analysis: %v", err)
	} else {
		fmt.Printf("感情: %s (信頼度: %.2f)\n", sentimentResult.Sentiment, sentimentResult.Confidence)
		fmt.Printf("キーワード: %v\n", sentimentResult.Keywords)
	}

	// === デモ4: 構造化出力（ジョーク生成）===
	fmt.Println("\n=== デモ4: 構造化出力（ジョーク生成）===")
	jokeResult, _, err := genkitsdk.GenerateData[JokeOutput](ctx, a.g,
		ai.WithModel(a.model),
		ai.WithPrompt(`
あなたはコメディアンです。「プログラミング」についてのジョークを考えてください。
必ずJSON形式で以下のフィールドを含めてください：
- setup: ジョークの前振り
- punchline: ジョークのオチ
- rating: 面白さの自己評価（1-10）
`),
	)
	if err != nil {
		log.Printf("Failed to generate joke: %v", err)
	} else {
		fmt.Printf("前振り: %s\n", jokeResult.Setup)
		fmt.Printf("オチ: %s\n", jokeResult.Punchline)
		fmt.Printf("自己評価: %d/10\n", jokeResult.Rating)
	}

	// === デモ5: RAG（知識検索型生成）===
	if a.useGemini {
		fmt.Println("\n=== デモ5: RAG（知識検索型生成）===")

		// ドキュメントをインデックス
		if err := a.indexDocuments(ctx); err != nil {
			log.Printf("Failed to index documents: %v", err)
		} else {
			// RAGで質問に回答
			questions := []string{
				"Genkitとは何ですか？",
				"Tool Callingで何ができますか？",
				"構造化出力の利点は？",
			}

			for _, q := range questions {
				fmt.Printf("\n質問: %s\n", q)
				answer, err := a.ragQuery(ctx, q)
				if err != nil {
					log.Printf("RAG query failed: %v", err)
				} else {
					fmt.Printf("回答: %s\n", answer)
				}
			}
		}
	} else {
		fmt.Println("\n=== デモ5: RAG（スキップ - Gemini APIが必要）===")
	}

	fmt.Println("\n=== 完了 ===")
	fmt.Println("\nREST APIモードで起動するには: MODE=server go run main.go")
}
