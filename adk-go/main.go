package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/full"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
	"google.golang.org/genai"
)

// ツールの入出力型
type SearchInput struct {
	Query string `json:"query" description:"検索クエリ"`
}

type SearchOutput struct {
	Result string `json:"result" description:"検索結果"`
}

// AIナレッジ検索ツール
func searchKnowledge(ctx tool.Context, input SearchInput) (SearchOutput, error) {
	// TODO: MCPクライアントでmcp/ai_knowledge/を呼び出す
	return SearchOutput{
		Result: fmt.Sprintf("AI知識検索結果: '%s' に関する情報を取得しました", input.Query),
	}, nil
}

// Web検索ツール
func webSearch(ctx tool.Context, input SearchInput) (SearchOutput, error) {
	// TODO: MCPクライアントでmcp/external_api/を呼び出す
	return SearchOutput{
		Result: fmt.Sprintf("Web検索結果: '%s' の最新情報を取得しました", input.Query),
	}, nil
}

func main() {
	ctx := context.Background()

	apiKey := os.Getenv("GOOGLE_API_KEY")
	if apiKey == "" {
		log.Fatal("GOOGLE_API_KEY が設定されていません")
	}

	// Geminiモデルの初期化
	model, err := gemini.NewModel(ctx, "gemini-2.0-flash", &genai.ClientConfig{
		APIKey: apiKey,
	})
	if err != nil {
		log.Fatalf("モデル初期化エラー: %v", err)
	}

	// ツールの定義
	knowledgeTool, err := functiontool.New(functiontool.Config{
		Name:        "search_knowledge",
		Description: "AIエンジニアリングに関する知識を検索します。RAG、評価パイプライン、プロンプトエンジニアリング、エージェント設計、ファインチューニングなどのトピック。",
	}, searchKnowledge)
	if err != nil {
		log.Fatalf("ツール作成エラー: %v", err)
	}

	webSearchTool, err := functiontool.New(functiontool.Config{
		Name:        "web_search",
		Description: "インターネットで最新情報を検索します。2026年の最新動向、技術トレンド、ニュースなど。",
	}, webSearch)
	if err != nil {
		log.Fatalf("ツール作成エラー: %v", err)
	}

	// エージェントの定義
	myAgent, err := llmagent.New(llmagent.Config{
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
	if err != nil {
		log.Fatalf("エージェント作成エラー: %v", err)
	}

	// エージェント実行（CLI or WebUI）
	l := full.NewLauncher()
	config := &launcher.Config{
		AgentLoader: agent.NewSingleLoader(myAgent),
	}

	if err := l.Execute(ctx, config, os.Args[1:]); err != nil {
		log.Fatalf("エージェント実行エラー: %v", err)
	}
}
