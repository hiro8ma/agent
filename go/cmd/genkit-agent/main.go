package main

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"os"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"github.com/firebase/genkit/go/plugins/mcp"

	"github.com/hiro8ma/agent/go/internal/action"
	actionclient "github.com/hiro8ma/agent/go/internal/action/client"
	"github.com/hiro8ma/agent/go/internal/agentcore"
	conversation "github.com/hiro8ma/agent/go/internal/conversation/client"
	"github.com/hiro8ma/agent/go/internal/genkitagent/agent"
	"github.com/hiro8ma/agent/go/internal/genkitagent/backend"
	knowledgeclient "github.com/hiro8ma/agent/go/internal/knowledge/client"
	"github.com/hiro8ma/agent/go/internal/lib/libconnect"
	"github.com/hiro8ma/agent/go/internal/lib/liblog"
	"github.com/hiro8ma/agent/go/internal/lib/libotel"
	"github.com/hiro8ma/agent/go/internal/lib/libserver"
)

type config struct {
	port            string
	vertexProjectID string
	vertexLocation  string
	geminiAPIKey    string // Vertex AI の代わりに Gemini Developer API を使う場合
	defaultModel    string
	mcpServerURL    string // 空なら MCP 連携なし（Streamable HTTP の URL）
	skillsDir       string // 空なら Agent Skills なし（SKILL.md を持つディレクトリの親）
	budget          agentcore.BudgetLimits
}

func loadConfig() (*config, error) {
	c := &config{
		port:            envOr("PORT", "19910"),
		vertexProjectID: os.Getenv("VERTEX_PROJECT_ID"),
		vertexLocation:  envOr("VERTEX_LOCATION", "asia-northeast1"),
		geminiAPIKey:    envOr("GEMINI_API_KEY", os.Getenv("GOOGLE_API_KEY")),
		mcpServerURL:    os.Getenv("MCP_SERVER_URL"),
		skillsDir:       os.Getenv("SKILLS_DIR"),
		budget:          agentcore.BudgetLimitsFromEnv(),
	}

	// バックエンドは Vertex AI と Gemini Developer API の 2 択。
	// モデル名の接頭辞がプラグインごとに異なるため、既定値もバックエンドで変える。
	switch {
	case c.vertexProjectID != "":
		c.defaultModel = envOr("DEFAULT_MODEL", "vertexai/gemini-3.8-flash")
	case c.geminiAPIKey != "":
		c.defaultModel = envOr("DEFAULT_MODEL", "googleai/gemini-3.8-flash")
	default:
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
		slog.String("defaultModel", c.defaultModel),
		slog.String("mcpServerURL", c.mcpServerURL),
		slog.String("skillsDir", c.skillsDir),
	)
}

func (c *config) String() string {
	return fmt.Sprintf("config{port:%s model:%s vertexProjectID:%s geminiAPIKey:%s}",
		c.port, c.defaultModel, c.vertexProjectID, agentcore.MaskSecret(c.geminiAPIKey))
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func main() {
	logger, err := liblog.InitFromEnv("genkit-agent")
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
	shutdown, err := libotel.Setup(ctx, "genkit-agent")
	if err != nil {
		return err
	}
	cfg, err := loadConfig()
	if err != nil {
		return err
	}

	var plugin api.Plugin
	if cfg.vertexProjectID != "" {
		plugin = &googlegenai.VertexAI{ProjectID: cfg.vertexProjectID, Location: cfg.vertexLocation}
	} else {
		plugin = &googlegenai.GoogleAI{APIKey: cfg.geminiAPIKey}
	}

	g := genkit.Init(ctx,
		genkit.WithPlugins(plugin),
		genkit.WithDefaultModel(cfg.defaultModel),
	)

	sessions, where, err := conversation.FromEnv()
	if err != nil {
		return fmt.Errorf("conversation store: %w", err)
	}
	logger.Info("conversation store", "where", where)

	gate, gateWhere := actionclient.FromEnv()
	logger.Info("action gate", "where", gateWhere)

	mcpTools, err := loadMCPTools(ctx, g, cfg.mcpServerURL)
	if err != nil {
		return err
	}
	if len(mcpTools) > 0 {
		logger.Info("mcp tools loaded", "count", len(mcpTools), "server", cfg.mcpServerURL)
	}

	orders := backend.NewInMemoryOrders()
	geo := backend.NewInMemoryGeo()
	kn, knWhere := knowledgeclient.FromEnv()
	logger.Info("knowledge searcher", "where", knWhere)

	var skillPaths []string
	if cfg.skillsDir != "" {
		skillPaths = []string{cfg.skillsDir}
		logger.Info("agent skills enabled", "dir", cfg.skillsDir)
	}

	// 部署別エージェント。system prompt とツールの組み合わせだけが違う
	research := agent.New(g, agent.Definition{
		ID:          "research",
		Description: "技術調査エージェント。社内ナレッジと外部ツールで調査に答える",
		SystemPrompt: "あなたは技術調査を支援するアシスタントです。" +
			"社内ナレッジ（search_knowledge）と利用可能なツールで事実を集め、出典がわかる形で簡潔に日本語で回答してください。" +
			"取得できなかった情報を推測で補わないでください。",
		Tools:      append(defineToolRefs(g, kn), mcpTools...),
		SkillPaths: skillPaths,
	})
	operations := agent.New(g, agent.Definition{
		ID:          "operations",
		Description: "申請処理エージェント。注文の照会と変更申請を扱う",
		SystemPrompt: "あなたは申請処理を支援するアシスタントです。" +
			"注文やエリアの質問にはツールで事実を取得して簡潔に日本語で回答してください。" +
			"変更系の操作は承認が必要です。承認待ちになった場合はその旨をユーザーに伝えてください。" +
			"取得できなかった情報を推測で補わないでください。",
		Tools:      agent.DefineOperationsTools(g, orders, geo, gate),
		SkillPaths: skillPaths,
	})

	registry := agentcore.NewRegistry(research, operations)
	executor := action.Executor{Gate: gate, Orders: orders}

	core := agentcore.NewHandler(registry, sessions, executor, logger)
	if cfg.budget.Enabled() {
		core = core.WithBudget(agentcore.NewBudgetTracker(cfg.budget))
		logger.Info("token budget enabled", "sessionTokens", cfg.budget.SessionTokens, "totalTokens", cfg.budget.TotalTokens)
	}

	mux := http.NewServeMux()
	mux.Handle(agentcore.NewConnectHandler(core, libconnect.HeaderAuthenticator))

	logger.Info("starting agent server", "port", cfg.port, "model", cfg.defaultModel, "agents", len(registry.List()))
	return libserver.Serve(ctx, logger, ":"+cfg.port, mux, shutdown)
}

// defineToolRefs は research エージェント用のツール群。
func defineToolRefs(g *genkit.Genkit, kn agent.KnowledgeSearcher) []ai.ToolRef {
	return agent.DefineResearchTools(g, kn)
}

// loadMCPTools は MCP サーバー（社内システム相当）のツールを取り込む。
func loadMCPTools(ctx context.Context, g *genkit.Genkit, url string) ([]ai.ToolRef, error) {
	if url == "" {
		return nil, nil
	}
	client, err := mcp.NewGenkitMCPClient(mcp.MCPClientOptions{
		Name:           "internal-systems",
		StreamableHTTP: &mcp.StreamableHTTPConfig{BaseURL: url},
	})
	if err != nil {
		return nil, fmt.Errorf("connect mcp server %s: %w", url, err)
	}
	tools, err := client.GetActiveTools(ctx, g)
	if err != nil {
		return nil, fmt.Errorf("load mcp tools: %w", err)
	}
	refs := make([]ai.ToolRef, len(tools))
	for i, t := range tools {
		refs[i] = t
	}
	return refs, nil
}
