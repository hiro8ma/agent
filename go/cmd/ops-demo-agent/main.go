// ops-demo-agent は、運用の画面（ダッシュボードとアラート）を確かめるための AgentService。
//
// adk-agent と同じ組み立てで、モデルだけを台本に替える。本物のモデルは呼ばない。本番では使わない。
//
// 環境変数
//
//	PORT              待ち受けるポート（既定 19912）
//	KNOWLEDGE_URL / ACTION_URL / CONVERSATION_URL  下流のサービス
//	GENAI_PRICES / GENAI_DAILY_BUDGET_USD         費用の計算（genaimetrics）
package main

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"time"

	"go.opentelemetry.io/otel"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/plugin"

	"github.com/hiro8ma/agent/go/internal/action"
	actionclient "github.com/hiro8ma/agent/go/internal/action/client"
	"github.com/hiro8ma/agent/go/internal/adk/llmretry"
	"github.com/hiro8ma/agent/go/internal/adk/toolgov"
	"github.com/hiro8ma/agent/go/internal/adkagent"
	"github.com/hiro8ma/agent/go/internal/agentcore"
	conversation "github.com/hiro8ma/agent/go/internal/conversation/client"
	"github.com/hiro8ma/agent/go/internal/genkitagent/backend"
	"github.com/hiro8ma/agent/go/internal/guardrail"
	knowledgeclient "github.com/hiro8ma/agent/go/internal/knowledge/client"
	"github.com/hiro8ma/agent/go/internal/lib/genaimetrics"
	"github.com/hiro8ma/agent/go/internal/lib/genaimetrics/adkmetrics"
	"github.com/hiro8ma/agent/go/internal/lib/libconnect"
	"github.com/hiro8ma/agent/go/internal/lib/liblog"
	"github.com/hiro8ma/agent/go/internal/lib/libotel"
	"github.com/hiro8ma/agent/go/internal/lib/libserver"
)

const service = "ops-demo-agent"

// policy は、運用の画面で拒否を見せるため operations から resolve_area_names を外す。
var policy = toolgov.Policy{
	"research":   {"search_knowledge"},
	"operations": {"get_order", action.ToolUpdatePaymentMethod},
}

func main() {
	logger, err := liblog.InitFromEnv(service)
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
	shutdown, err := libotel.Setup(ctx, service)
	if err != nil {
		return err
	}
	cfg, err := genaimetrics.ConfigFromEnv()
	if err != nil {
		return err
	}
	rec, err := genaimetrics.New(otel.GetMeterProvider(), cfg)
	if err != nil {
		return err
	}
	metrics, err := adkmetrics.Plugin(rec, adkmetrics.WithEscalation(func(_ string, r map[string]any) bool {
		return r["status"] == action.StatusPending
	}))
	if err != nil {
		return err
	}
	gov, err := toolgov.New(policy, &toolgov.Memory{})
	if err != nil {
		return err
	}
	plugins := []*plugin.Plugin{gov, metrics}

	guards := guardrail.NewLog()
	guards.OnBlock(func(v guardrail.Verdict) { rec.GuardrailBlocked(context.Background(), v.Stage, v.Rule) })
	beforeModel := []llmagent.BeforeModelCallback{guardrail.DetectInjection(guards, guardrail.InjectionPatterns, "その依頼には対応できません")}

	retry := llmretry.DefaultPolicy()
	retry.BaseDelay = 200 * time.Millisecond
	retry.OnRetry = rec.Retry

	kn, _ := knowledgeclient.FromEnv()
	gate, _ := actionclient.FromEnv()
	orders := backend.NewInMemoryOrders()
	researchTools, err := adkagent.ResearchTools(kn)
	if err != nil {
		return err
	}
	operationsTools, err := adkagent.OperationsTools(orders, backend.NewInMemoryGeo(), gate)
	if err != nil {
		return err
	}
	research, err := adkagent.New(adkagent.Definition{
		ID: "research", Description: "調査", Instruction: "x",
		Model:   llmretry.Wrap(newScripted("gemini-3.8-flash", false, 1), retry),
		Tools:   researchTools,
		Plugins: plugins, BeforeModelCallbacks: beforeModel,
	})
	if err != nil {
		return fmt.Errorf("research agent: %w", err)
	}
	operations, err := adkagent.New(adkagent.Definition{
		ID: "operations", Description: "申請処理", Instruction: "x",
		Model:   llmretry.Wrap(newScripted("gemini-3.8-pro", true, 2), retry),
		Tools:   operationsTools,
		Plugins: plugins, BeforeModelCallbacks: beforeModel,
	})
	if err != nil {
		return fmt.Errorf("operations agent: %w", err)
	}

	sessions, _, err := conversation.FromEnv()
	if err != nil {
		return err
	}
	core := agentcore.NewHandler(agentcore.NewRegistry(research, operations), sessions, action.Executor{Gate: gate, Orders: orders}, logger)
	mux := http.NewServeMux()
	mux.Handle(agentcore.NewConnectHandler(core, libconnect.HeaderAuthenticator))
	return libserver.Serve(ctx, logger, ":"+envOr("PORT", "19912"), mux, shutdown)
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
