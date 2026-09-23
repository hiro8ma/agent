// Package main は旅行プランナーのエントリ。
//
// Python 版は `adk run travel_planner` でディレクトリから root_agent を探す。
// こちらは組み立てた木を launcher へ明示的に渡す。
package main

import (
	"context"
	"log"
	"os"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/cmd/launcher"
	"google.golang.org/adk/v2/cmd/launcher/full"
	"google.golang.org/adk/v2/model/gemini"
	"google.golang.org/genai"

	"github.com/hiro8ma/agent/go/internal/adk/llmretry"
	"github.com/hiro8ma/agent/go/internal/adk/travelplanner"
)

func main() {
	ctx := context.Background()

	apiKey := os.Getenv("GOOGLE_API_KEY")
	if apiKey == "" {
		apiKey = os.Getenv("GEMINI_API_KEY")
	}
	if apiKey == "" {
		log.Fatal("GOOGLE_API_KEY または GEMINI_API_KEY を設定してください")
	}

	m, err := gemini.NewModel(ctx, travelplanner.ModelName, &genai.ClientConfig{APIKey: apiKey})
	if err != nil {
		log.Fatalf("failed to create model: %v", err)
	}
	// 無料枠は 1 分あたり 5 回で、調査の 3 並列だけで越える。429 の待ち時間に従って再試行する
	a, err := travelplanner.NewWithModel(llmretry.Wrap(m, llmretry.DefaultPolicy()))
	if err != nil {
		log.Fatalf("failed to build agent: %v", err)
	}

	l := full.NewLauncher()
	cfg := &launcher.Config{AgentLoader: agent.NewSingleLoader(a)}
	if err := l.Execute(ctx, cfg, os.Args[1:]); err != nil {
		log.Fatalf("run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}
