package main

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"os"

	"cloud.google.com/go/firestore"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"github.com/firebase/genkit/go/plugins/mcp"
	"golang.org/x/net/http2"
	"golang.org/x/net/http2/h2c"

	"github.com/hiro8ma/agent/go/gen/agent/v1/agentv1connect"
	"github.com/hiro8ma/agent/go/internal/agentcore"
	"github.com/hiro8ma/agent/go/internal/genkitagent/agent"
	"github.com/hiro8ma/agent/go/internal/genkitagent/backend"
	"github.com/hiro8ma/agent/go/internal/genkitagent/knowledge"
	"github.com/hiro8ma/agent/go/internal/genkitagent/session"
)

type config struct {
	port             string
	vertexProjectID  string
	vertexLocation   string
	geminiAPIKey     string // Vertex AI の代わりに Gemini Developer API を使う場合
	defaultModel     string
	firestoreProject string // 空なら履歴・承認待ちはインメモリ
	mcpServerURL     string // 空なら MCP 連携なし（Streamable HTTP の URL）
	skillsDir        string // 空なら Agent Skills なし（SKILL.md を持つディレクトリの親）
	budget           agentcore.BudgetLimits
}

func loadConfig() (*config, error) {
	c := &config{
		port:             envOr("PORT", "19910"),
		vertexProjectID:  os.Getenv("VERTEX_PROJECT_ID"),
		vertexLocation:   envOr("VERTEX_LOCATION", "asia-northeast1"),
		geminiAPIKey:     envOr("GEMINI_API_KEY", os.Getenv("GOOGLE_API_KEY")),
		firestoreProject: os.Getenv("FIRESTORE_PROJECT_ID"),
		mcpServerURL:     os.Getenv("MCP_SERVER_URL"),
		skillsDir:        os.Getenv("SKILLS_DIR"),
		budget:           agentcore.BudgetLimitsFromEnv(),
	}

	// バックエンドは Vertex AI と Gemini Developer API の 2 択。
	// モデル名の接頭辞がプラグインごとに異なるため、既定値もバックエンドで変える。
	switch {
	case c.vertexProjectID != "":
		c.defaultModel = envOr("DEFAULT_MODEL", "vertexai/gemini-3.5-flash")
	case c.geminiAPIKey != "":
		c.defaultModel = envOr("DEFAULT_MODEL", "googleai/gemini-3.5-flash")
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
		slog.String("firestoreProject", c.firestoreProject),
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

	var sessions session.Store = session.NewInMemory()
	var pending agent.PendingStore = session.NewInMemoryPending()
	if cfg.firestoreProject != "" {
		client, err := firestore.NewClient(ctx, cfg.firestoreProject)
		if err != nil {
			return fmt.Errorf("new firestore client: %w", err)
		}
		defer func() { _ = client.Close() }()
		fs := session.NewFirestore(client)
		sessions = fs
		pending = fs
	}

	mcpTools, err := loadMCPTools(ctx, g, cfg.mcpServerURL)
	if err != nil {
		return err
	}
	if len(mcpTools) > 0 {
		logger.Info("mcp tools loaded", "count", len(mcpTools), "server", cfg.mcpServerURL)
	}

	orders := backend.NewInMemoryOrders()
	geo := backend.NewInMemoryGeo()
	kn := knowledge.NewInMemory()

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
		Tools:      agent.DefineOperationsTools(g, orders, geo, pending),
		SkillPaths: skillPaths,
	})

	registry := agentcore.NewRegistry(research, operations)
	executor := agent.NewExecutor(orders, pending)

	core := agentcore.NewHandler(registry, sessions, executor, logger)
	if cfg.budget.Enabled() {
		core = core.WithBudget(agentcore.NewBudgetTracker(cfg.budget))
		logger.Info("token budget enabled", "sessionTokens", cfg.budget.SessionTokens, "totalTokens", cfg.budget.TotalTokens)
	}

	mux := http.NewServeMux()
	path, handler := agentv1connect.NewAgentServiceHandler(core)
	mux.Handle(path, handler)

	logger.Info("starting agent server", "port", cfg.port, "model", cfg.defaultModel, "agents", len(registry.List()))
	return http.ListenAndServe(":"+cfg.port, h2c.NewHandler(mux, &http2.Server{}))
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
