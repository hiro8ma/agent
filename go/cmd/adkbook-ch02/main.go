// Package main は教材の第 2 章のエントリ。
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

	"github.com/hiro8ma/agent/go/internal/adkbook/chapter02"
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

	a, err := chapter02.New(ctx, apiKey)
	if err != nil {
		log.Fatalf("failed to build agent: %v", err)
	}

	l := full.NewLauncher()
	cfg := &launcher.Config{AgentLoader: agent.NewSingleLoader(a)}
	if err := l.Execute(ctx, cfg, os.Args[1:]); err != nil {
		log.Fatalf("run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}
