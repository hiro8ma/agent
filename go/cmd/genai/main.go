// Package main は genai 実装のエントリ。
// DI（依存性注入）はここで行う。
package main

import (
	"context"
	"log"
	"os"

	sdkgenai "google.golang.org/genai"

	"github.com/hiro8ma/agent/go/internal/agent"
	"github.com/hiro8ma/agent/go/internal/agent/usecase"
	"github.com/hiro8ma/agent/go/internal/genai"
	"github.com/hiro8ma/agent/go/internal/handler/cli"
	"github.com/hiro8ma/agent/go/internal/tool"
)

const modelName = "gemini-2.0-flash"

func main() {
	ctx := context.Background()

	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		apiKey = os.Getenv("GOOGLE_API_KEY")
	}
	if apiKey == "" {
		log.Fatal("GEMINI_API_KEY または GOOGLE_API_KEY を設定してください")
	}

	client, err := sdkgenai.NewClient(ctx, &sdkgenai.ClientConfig{
		APIKey:  apiKey,
		Backend: sdkgenai.BackendGeminiAPI,
	})
	if err != nil {
		log.Fatalf("client init %v", err)
	}

	// Adapter 層 LLM
	llm := genai.New(client, modelName)

	// Adapter 層 Tool Registry（3 実装共有）
	tools := tool.Default()

	// Usecase 層 Ask（SDK 非依存）
	ask := usecase.NewAsk(
		usecase.DefaultAskConfig(agent.SystemPrompt),
		llm,
		tools,
	)

	// Handler 層 CLI
	cli.NewDemo("genai", ask).Run(ctx)
}
