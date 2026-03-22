package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/google/adk-go/pkg/agent"
	"github.com/google/adk-go/pkg/model/gemini"
	"github.com/google/adk-go/pkg/tool"
)

// AIナレッジ検索ツール — mcp/ai_knowledge/ のMCPサーバーを呼び出す代わりに
// ここではシンプルなツール定義でReActパターンの雛形を作る
func searchKnowledge(ctx context.Context, query string) (string, error) {
	// TODO: MCPクライアントでmcp/ai_knowledge/を呼び出す
	return fmt.Sprintf("AI知識検索結果: '%s' に関する情報を取得しました", query), nil
}

func webSearch(ctx context.Context, query string) (string, error) {
	// TODO: MCPクライアントでmcp/external_api/を呼び出す
	return fmt.Sprintf("Web検索結果: '%s' の最新情報を取得しました", query), nil
}

func main() {
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		log.Fatal("GEMINI_API_KEY が設定されていません")
	}

	// Geminiモデルの初期化
	model := gemini.NewModel("gemini-2.0-flash", apiKey)

	// ツールの定義
	knowledgeTool := tool.NewFunction(
		"search_knowledge",
		"AIエンジニアリングに関する知識を検索します。RAG、評価パイプライン、プロンプトエンジニアリング、エージェント設計、ファインチューニングなどのトピック。",
		searchKnowledge,
	)

	webSearchTool := tool.NewFunction(
		"web_search",
		"インターネットで最新情報を検索します。2026年の最新動向、技術トレンド、ニュースなど。",
		webSearch,
	)

	// エージェントの定義
	myAgent := agent.New(agent.Config{
		Name:        "ai-engineering-agent",
		Description: "AIエンジニアリングの質問に答える自律エージェント",
		Instruction: `あなたはAIエンジニアリングの専門家エージェントです。

ユーザーの質問に対して、以下のプロセスで回答してください:

1. 質問を分析し、どのツールが必要か判断する
2. search_knowledge でAIエンジニアリングの知識を検索
3. 必要に応じて web_search で最新情報を補完
4. 取得した情報を統合して、実務に即した回答を生成

回答は技術的に正確で、具体例を含むようにしてください。`,
		Model: model,
		Tools: []tool.Tool{knowledgeTool, webSearchTool},
	})

	// エージェント実行
	ctx := context.Background()
	if err := myAgent.Run(ctx); err != nil {
		log.Fatalf("エージェント実行エラー: %v", err)
	}
}
