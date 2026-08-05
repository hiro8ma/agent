// adk-agent は AgentService（proto/agent/v1）の ADK 版サーバー。
// genkit 版（cmd/genkit-agent、PORT 19910）と同じ proto を実装し、cmd/genkit-ask で動作確認できる。
package main

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"os"

	"golang.org/x/net/http2"
	"golang.org/x/net/http2/h2c"
	"google.golang.org/adk/v2/model/gemini"
	"google.golang.org/genai"

	"github.com/hiro8ma/agent/go/gen/agent/v1/agentv1connect"
	"github.com/hiro8ma/agent/go/internal/adkagent"
	"github.com/hiro8ma/agent/go/internal/agentcore"
	"github.com/hiro8ma/agent/go/internal/genkitagent/backend"
	"github.com/hiro8ma/agent/go/internal/genkitagent/knowledge"
	"github.com/hiro8ma/agent/go/internal/genkitagent/session"
)

type config struct {
	port            string
	vertexProjectID string
	vertexLocation  string
	modelName       string
}

func loadConfig() (*config, error) {
	c := &config{
		port:            envOr("PORT", "19912"),
		vertexProjectID: os.Getenv("VERTEX_PROJECT_ID"),
		vertexLocation:  envOr("VERTEX_LOCATION", "asia-northeast1"),
		modelName:       envOr("DEFAULT_MODEL", "gemini-2.5-flash"),
	}
	if c.vertexProjectID == "" {
		return nil, fmt.Errorf("VERTEX_PROJECT_ID is required")
	}
	return c, nil
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

// noopExecutor は Phase 1 の仮実装。承認フローは Tool Confirmation API で Phase 2 に実装する。
type noopExecutor struct{}

func (noopExecutor) Execute(context.Context, string) (map[string]any, error) {
	return nil, agentcore.ErrPendingNotFound
}

func main() {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))
	if err := run(context.Background(), logger); err != nil {
		logger.Error("server exited", "error", err)
		os.Exit(1)
	}
}

func run(ctx context.Context, logger *slog.Logger) error {
	cfg, err := loadConfig()
	if err != nil {
		return err
	}

	m, err := gemini.NewModel(ctx, cfg.modelName, &genai.ClientConfig{
		Backend:  genai.BackendVertexAI,
		Project:  cfg.vertexProjectID,
		Location: cfg.vertexLocation,
	})
	if err != nil {
		return fmt.Errorf("gemini model init: %w", err)
	}

	orders := backend.NewInMemoryOrders()
	geo := backend.NewInMemoryGeo()
	kn := knowledge.NewInMemory()

	researchTools, err := adkagent.ResearchTools(kn)
	if err != nil {
		return fmt.Errorf("research tools: %w", err)
	}
	operationsTools, err := adkagent.OperationsTools(orders, geo)
	if err != nil {
		return fmt.Errorf("operations tools: %w", err)
	}

	research, err := adkagent.New(adkagent.Definition{
		ID:          "research",
		Description: "技術調査エージェント。社内ナレッジと外部ツールで調査に答える",
		Instruction: "あなたは技術調査を支援するアシスタントです。" +
			"社内ナレッジ（search_knowledge）と利用可能なツールで事実を集め、出典がわかる形で簡潔に日本語で回答してください。" +
			"取得できなかった情報を推測で補わないでください。",
		Model: m,
		Tools: researchTools,
	})
	if err != nil {
		return fmt.Errorf("research agent: %w", err)
	}
	operations, err := adkagent.New(adkagent.Definition{
		ID:          "operations",
		Description: "申請処理エージェント。注文の照会と変更申請を扱う",
		Instruction: "あなたは申請処理を支援するアシスタントです。" +
			"注文やエリアの質問にはツールで事実を取得して簡潔に日本語で回答してください。" +
			"変更系の操作は現在対応していません。依頼された場合はその旨を伝えてください。" +
			"取得できなかった情報を推測で補わないでください。",
		Model: m,
		Tools: operationsTools,
	})
	if err != nil {
		return fmt.Errorf("operations agent: %w", err)
	}

	registry := agentcore.NewRegistry(research, operations)

	mux := http.NewServeMux()
	path, handler := agentv1connect.NewAgentServiceHandler(
		agentcore.NewHandler(registry, session.NewInMemory(), noopExecutor{}, logger),
	)
	mux.Handle(path, handler)

	logger.Info("starting adk agent server", "port", cfg.port, "model", cfg.modelName, "agents", len(registry.List()))
	return http.ListenAndServe(":"+cfg.port, h2c.NewHandler(mux, &http2.Server{}))
}
