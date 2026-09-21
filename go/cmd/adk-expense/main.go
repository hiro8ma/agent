// Package main は経費精算のエージェントのエントリ。評価、ガードレール、HITL を組み合わせる。
//
//	go run ./cmd/adk-expense web -port 8080 api webui -api_server_address http://localhost:8080/api
//
// ADK の REST の /run は利用者 ID をリクエストの本文で受け、認証しない。
// 承認のツールは承認者だけが使える作りだが、利用者 ID を承認者の ID にして送れば誰でも承認者になれる。
// この起動口はローカルの確認用で、外に出すなら利用者を認証する前段を置く。
//
// 環境変数
//
//	EXPENSE_APPROVERS  承認者（カンマ区切り。既定 manager）
package main

import (
	"context"
	"log"
	"os"
	"strings"
	"time"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/cmd/launcher"
	"google.golang.org/adk/v2/cmd/launcher/full"
	"google.golang.org/adk/v2/model/gemini"
	"google.golang.org/genai"

	"github.com/hiro8ma/agent/go/internal/adk/expense"
	"github.com/hiro8ma/agent/go/internal/approval"
	"github.com/hiro8ma/agent/go/internal/guardrail"
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
	m, err := gemini.NewModel(ctx, expense.ModelName, &genai.ClientConfig{APIKey: apiKey})
	if err != nil {
		log.Fatalf("failed to build model: %v", err)
	}

	approvers := []string{"manager"}
	if v := os.Getenv("EXPENSE_APPROVERS"); v != "" {
		approvers = strings.Split(v, ",")
	}
	deps := expense.Deps{
		Store:     expense.NewStore(),
		Approvals: approval.NewService(expense.Rules(), approvers, 24*time.Hour),
		Approvers: approvers,
	}
	a, err := expense.NewAgent(m, deps, guardrail.NewLog())
	if err != nil {
		log.Fatalf("failed to build agent: %v", err)
	}

	l := full.NewLauncher()
	if err := l.Execute(ctx, &launcher.Config{AgentLoader: agent.NewSingleLoader(a)}, os.Args[1:]); err != nil {
		log.Fatalf("run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}
