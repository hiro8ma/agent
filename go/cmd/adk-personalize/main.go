// Package main は利用者ごとに答え方を変える技術アシスタントのエントリ。
//
// Session の保管先は .env の ADK_SESSION_BACKEND で選ぶ。
// 輪郭は user: の State に残るので、database を選べばプロセスを越えて引き継がれる。
package main

import (
	"context"
	"log"
	"os"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/cmd/launcher"
	"google.golang.org/adk/v2/cmd/launcher/full"
	"google.golang.org/adk/v2/model/gemini"

	"github.com/hiro8ma/agent/go/internal/adk/personalize"
	"github.com/hiro8ma/agent/go/internal/adk/repository"
)

const modelName = "gemini-3.8-flash"

func main() {
	ctx := context.Background()

	apiKey := os.Getenv("GOOGLE_API_KEY")
	if apiKey == "" {
		apiKey = os.Getenv("GEMINI_API_KEY")
	}
	if apiKey == "" {
		log.Fatal("GOOGLE_API_KEY または GEMINI_API_KEY を設定してください")
	}

	m, err := gemini.NewModel(ctx, modelName, &genai.ClientConfig{APIKey: apiKey})
	if err != nil {
		log.Fatalf("failed to build model: %v", err)
	}
	a, err := personalize.NewAgent(m)
	if err != nil {
		log.Fatalf("failed to build agent: %v", err)
	}

	sessions, backend, err := repository.NewSessionsFromEnv(ctx)
	if err != nil {
		log.Fatalf("failed to build session service: %v", err)
	}
	log.Printf("session backend: %s（永続化 %v）", backend, backend.Persistent())

	l := full.NewLauncher()
	cfg := &launcher.Config{AgentLoader: agent.NewSingleLoader(a), SessionService: sessions}
	if err := l.Execute(ctx, cfg, os.Args[1:]); err != nil {
		log.Fatalf("run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}
