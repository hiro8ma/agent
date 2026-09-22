// adk-agent は AgentService（proto/agent/v1）の ADK 版サーバー。
// genkit 版（cmd/genkit-agent、PORT 19910）と同じ proto を実装し、cmd/genkit-ask で動作確認できる。
package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"

	"go.opentelemetry.io/otel"
	"google.golang.org/adk/v2/model/gemini"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/genai"

	"github.com/hiro8ma/agent/go/internal/a2aserve"
	"github.com/hiro8ma/agent/go/internal/action"
	actionclient "github.com/hiro8ma/agent/go/internal/action/client"
	"github.com/hiro8ma/agent/go/internal/adk/a2ainterop"
	"github.com/hiro8ma/agent/go/internal/adk/llmretry"
	"github.com/hiro8ma/agent/go/internal/adkagent"
	"github.com/hiro8ma/agent/go/internal/agentcore"
	conversation "github.com/hiro8ma/agent/go/internal/conversation/client"
	"github.com/hiro8ma/agent/go/internal/genkitagent/backend"
	knowledgeclient "github.com/hiro8ma/agent/go/internal/knowledge/client"
	"github.com/hiro8ma/agent/go/internal/lib/genaimetrics"
	"github.com/hiro8ma/agent/go/internal/lib/genaimetrics/adkmetrics"
	"github.com/hiro8ma/agent/go/internal/lib/libconnect"
	"github.com/hiro8ma/agent/go/internal/lib/liblog"
	"github.com/hiro8ma/agent/go/internal/lib/libotel"
	"github.com/hiro8ma/agent/go/internal/lib/libserver"
	"github.com/hiro8ma/agent/go/internal/toolscope"
)

type config struct {
	port            string
	vertexProjectID string
	vertexLocation  string
	geminiAPIKey    string // Vertex AI の代わりに Gemini Developer API を使う場合
	modelName       string
	budget          agentcore.BudgetLimits
}

func loadConfig() (*config, error) {
	c := &config{
		port:            envOr("PORT", "19912"),
		vertexProjectID: os.Getenv("VERTEX_PROJECT_ID"),
		vertexLocation:  envOr("VERTEX_LOCATION", "asia-northeast1"),
		geminiAPIKey:    envOr("GEMINI_API_KEY", os.Getenv("GOOGLE_API_KEY")),
		modelName:       envOr("DEFAULT_MODEL", "gemini-3.8-flash"),
		budget:          agentcore.BudgetLimitsFromEnv(),
	}
	if c.vertexProjectID == "" && c.geminiAPIKey == "" {
		return nil, fmt.Errorf("VERTEX_PROJECT_ID または GEMINI_API_KEY が必要です")
	}
	return c, nil
}

// LogValue は config をログに出したときにキーが平文で出ることを防ぐ。
// slog は fmt.Stringer ではなく slog.LogValuer を見るため、両方を実装しておく。
func (c *config) LogValue() slog.Value {
	return slog.GroupValue(
		slog.String("port", c.port),
		slog.String("vertexProjectID", c.vertexProjectID),
		slog.String("vertexLocation", c.vertexLocation),
		slog.String("geminiAPIKey", agentcore.MaskSecret(c.geminiAPIKey)),
		slog.String("modelName", c.modelName),
	)
}

func (c *config) String() string {
	return fmt.Sprintf("config{port:%s model:%s vertexProjectID:%s geminiAPIKey:%s}",
		c.port, c.modelName, c.vertexProjectID, agentcore.MaskSecret(c.geminiAPIKey))
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func main() {
	logger, err := liblog.InitFromEnv("adk-agent")
	if err != nil {
		slog.Error("ロガーを作れない", "error", err)
		os.Exit(1)
	}
	if err := run(context.Background(), logger); err != nil {
		logger.Error("server exited", "error", err)
		os.Exit(1)
	}
}

func run(ctx context.Context, logger *slog.Logger) error {
	shutdown, err := libotel.Setup(ctx, "adk-agent")
	if err != nil {
		return err
	}
	cfg, err := loadConfig()
	if err != nil {
		return err
	}

	clientCfg := &genai.ClientConfig{
		Backend:  genai.BackendVertexAI,
		Project:  cfg.vertexProjectID,
		Location: cfg.vertexLocation,
	}
	if cfg.vertexProjectID == "" {
		clientCfg = &genai.ClientConfig{Backend: genai.BackendGeminiAPI, APIKey: cfg.geminiAPIKey}
	}

	gm, err := gemini.NewModel(ctx, cfg.modelName, clientCfg)
	if err != nil {
		return fmt.Errorf("gemini model init: %w", err)
	}
	metricsCfg, err := genaimetrics.ConfigFromEnv()
	if err != nil {
		return err
	}
	rec, err := genaimetrics.New(otel.GetMeterProvider(), metricsCfg)
	if err != nil {
		return err
	}
	metrics, err := adkmetrics.Plugin(rec, adkmetrics.WithEscalation(func(_ string, r map[string]any) bool {
		return r["status"] == action.StatusPending
	}))
	if err != nil {
		return err
	}
	retry := llmretry.DefaultPolicy()
	retry.OnRetry = rec.Retry
	m := llmretry.Wrap(gm, retry)
	plugins := []*plugin.Plugin{metrics}

	orders := backend.NewInMemoryOrders()
	geo := backend.NewInMemoryGeo()
	kn, knWhere := knowledgeclient.FromEnv()
	logger.Info("knowledge searcher", "where", knWhere)

	researchTools, err := adkagent.ResearchTools(kn)
	if err != nil {
		return fmt.Errorf("research tools: %w", err)
	}
	gate, gateWhere := actionclient.FromEnv()
	logger.Info("action gate", "where", gateWhere)
	operationsTools, err := adkagent.OperationsTools(orders, geo, gate)
	if err != nil {
		return fmt.Errorf("operations tools: %w", err)
	}

	research, err := adkagent.New(adkagent.Definition{
		ID:          "research",
		Description: "技術調査エージェント。社内ナレッジと外部ツールで調査に答える",
		Instruction: "あなたは技術調査を支援するアシスタントです。" +
			"社内ナレッジ（search_knowledge）と利用可能なツールで事実を集め、出典がわかる形で簡潔に日本語で回答してください。" +
			"取得できなかった情報を推測で補わないでください。",
		Model:   m,
		Tools:   researchTools,
		Plugins: plugins,
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
		Model:   m,
		Tools:   operationsTools,
		Plugins: plugins,
	})
	if err != nil {
		return fmt.Errorf("operations agent: %w", err)
	}

	registry := agentcore.NewRegistry(research, operations)

	sessions, where, err := conversation.FromEnv()
	if err != nil {
		return fmt.Errorf("conversation store: %w", err)
	}
	logger.Info("conversation store", "where", where)

	core := agentcore.NewHandler(registry, sessions, action.Executor{Gate: gate, Orders: orders}, logger)
	if cfg.budget.Enabled() {
		core = core.WithBudget(agentcore.NewBudgetTracker(cfg.budget))
		logger.Info("token budget enabled", "sessionTokens", cfg.budget.SessionTokens, "totalTokens", cfg.budget.TotalTokens)
	}

	if err := serveA2A(ctx, logger, map[string]*adkagent.Agent{"research": research, "operations": operations}); err != nil {
		return err
	}

	mux := http.NewServeMux()
	mux.Handle(agentcore.NewConnectHandler(core, libconnect.HeaderAuthenticator))

	logger.Info("starting adk agent server", "port", cfg.port, "model", cfg.modelName, "agents", len(registry.List()))
	return libserver.Serve(ctx, logger, ":"+cfg.port, mux, shutdown)
}

// serveA2A は A2A_ADDR があれば、選んだエージェントを 2 要素で守った A2A の口で公開する。
func serveA2A(ctx context.Context, logger *slog.Logger, agents map[string]*adkagent.Agent) error {
	cfg, enabled, err := a2aserve.ConfigFromEnv()
	if !enabled || err != nil {
		return err
	}
	a, ok := agents[cfg.AgentID]
	if !ok {
		return fmt.Errorf("A2A_AGENT %q は無い", cfg.AgentID)
	}
	scope, err := toolscope.ADKPlugin(a2aserve.Policy)
	if err != nil {
		return err
	}
	v, err := a2aserve.Verifier(ctx, cfg)
	if err != nil {
		return err
	}
	info := a.Info()
	h := a2ainterop.NewHandler(a.ADK(), a2aserve.Card(cfg, info.ID, info.Description), cfg.BaseURL, append(a.Plugins(), scope)...)
	go func() {
		if err := a2aserve.Serve(ctx, logger, cfg, a2aserve.Protect(h, v, cfg)); err != nil && !errors.Is(err, http.ErrServerClosed) {
			logger.ErrorContext(ctx, "a2a server exited", "error", err)
		}
	}()
	return nil
}
