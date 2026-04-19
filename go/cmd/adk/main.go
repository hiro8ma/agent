// Package main は adk 実装のエントリ。
package main

import (
	"context"
	"log"
	"os"

	"github.com/hiro8ma/agent/go/internal/adk"
	"github.com/hiro8ma/agent/go/internal/agent"
	"github.com/hiro8ma/agent/go/internal/handler/cli"
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

	ask, err := adk.New(ctx, adk.Config{
		APIKey:       apiKey,
		ModelName:    modelName,
		SystemPrompt: agent.SystemPrompt,
		AgentName:    "comparison-agent",
	})
	if err != nil {
		log.Fatalf("adk init %v", err)
	}

	cli.NewDemo("adk", ask).Run(ctx)
}
