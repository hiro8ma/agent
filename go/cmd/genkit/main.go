// Package main は genkit 実装のエントリ。
package main

import (
	"context"
	"log"
	"os"

	"github.com/firebase/genkit/go/ai"
	genkitsdk "github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"

	"github.com/hiro8ma/agent/go/internal/agent"
	"github.com/hiro8ma/agent/go/internal/genkit"
	"github.com/hiro8ma/agent/go/internal/handler/cli"
	"github.com/hiro8ma/agent/go/internal/tool"
)

const modelName = "gemini-2.0-flash"

func main() {
	ctx := context.Background()

	if os.Getenv("GEMINI_API_KEY") == "" && os.Getenv("GOOGLE_API_KEY") == "" {
		log.Fatal("GEMINI_API_KEY または GOOGLE_API_KEY を設定してください")
	}

	g := genkitsdk.Init(ctx, genkitsdk.WithPlugins(&googlegenai.GoogleAI{}))

	var model ai.Model = googlegenai.GoogleAIModel(g, modelName)

	specs := []genkit.ToolSpec{
		{Schema: tool.CalculatorSchema, Handler: tool.Calculator},
		{Schema: tool.SearchKnowledgeSchema, Handler: tool.SearchKnowledge},
	}

	ask := genkit.New(g, model, agent.SystemPrompt, specs)

	cli.NewDemo("genkit", ask).Run(ctx)
}
