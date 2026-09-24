// Package main は旅行プランナーを TravelPlannerService（ConnectRPC）として公開する起動口。
//
// -impl で ADK 版か Genkit 版を選ぶ。どちらも同じハンドラーで公開する。
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"

	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"google.golang.org/adk/v2/model/gemini"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/genai"

	"github.com/hiro8ma/agent/go/internal/adk/callguard"
	"github.com/hiro8ma/agent/go/internal/adk/llmretry"
	adktravel "github.com/hiro8ma/agent/go/internal/adk/travelplanner"
	genkittravel "github.com/hiro8ma/agent/go/internal/genkit/travelplanner"
	"github.com/hiro8ma/agent/go/internal/lib/libconnect"
	"github.com/hiro8ma/agent/go/internal/lib/liblog"
	"github.com/hiro8ma/agent/go/internal/lib/libotel"
	"github.com/hiro8ma/agent/go/internal/lib/libserver"
	"github.com/hiro8ma/agent/go/internal/travel"
	"github.com/hiro8ma/agent/go/internal/travel/connecthandler"
)

func main() {
	impl := flag.String("impl", "adk", "旅行プランナーの実装（adk / genkit）")
	port := flag.String("port", "19950", "待ち受けるポート")
	flag.Parse()

	logger, err := liblog.InitFromEnv("travelplanner-server")
	if err != nil {
		slog.Error("ロガーを作れない", "error", err)
		os.Exit(1)
	}
	if err := run(context.Background(), logger, *impl, *port); err != nil {
		logger.Error("server exited", "error", err)
		os.Exit(1)
	}
}

func run(ctx context.Context, logger *slog.Logger, impl, port string) error {
	var (
		planner travel.Planner
		err     error
	)
	skillsDir := os.Getenv("TRAVEL_SKILLS_DIR")
	switch impl {
	case "adk":
		planner, err = newADKPlanner(ctx, modelOr(adktravel.ModelName), skillsDir)
	case "genkit":
		planner, err = newGenkitPlanner(ctx, modelOr(genkittravel.ModelName), skillsDir)
	default:
		err = fmt.Errorf("-impl は adk か genkit を指定してください: %q", impl)
	}
	if err != nil {
		return err
	}

	shutdown, err := libotel.Setup(ctx, "travelplanner-server")
	if err != nil {
		return err
	}
	mux := http.NewServeMux()
	mux.Handle(connecthandler.Route(connecthandler.New(planner), libconnect.HeaderAuthenticator))
	logger.Info("travel planner", "impl", impl, "skillsDir", skillsDir)
	return libserver.Serve(ctx, logger, ":"+port, mux, shutdown)
}

func newADKPlanner(ctx context.Context, modelName, skillsDir string) (travel.Planner, error) {
	apiKey := os.Getenv("GOOGLE_API_KEY")
	if apiKey == "" {
		apiKey = os.Getenv("GEMINI_API_KEY")
	}
	if apiKey == "" {
		return nil, errors.New("GOOGLE_API_KEY または GEMINI_API_KEY を設定してください")
	}
	m, err := gemini.NewModel(ctx, modelName, &genai.ClientConfig{APIKey: apiKey})
	if err != nil {
		return nil, fmt.Errorf("create model: %w", err)
	}
	// 通常の 1 回は調査 3 つが各 2 回、日程と予算が各 1 回で、モデルを 8 回呼ぶ
	guard, err := callguard.New(callguard.Limits{MaxLLMCalls: 20, MaxSameToolCalls: 3})
	if err != nil {
		return nil, fmt.Errorf("build callguard: %w", err)
	}
	var skills tool.Toolset
	if skillsDir != "" {
		if skills, err = adktravel.NewSkillToolset(ctx, skillsDir); err != nil {
			return nil, err
		}
	}
	// 無料枠は 1 分あたり 5 回で、調査の 3 並列だけで越える。429 の待ち時間に従って再試行する
	return adktravel.NewPlanner(llmretry.Wrap(m, llmretry.DefaultPolicy()), skills, guard)
}

func newGenkitPlanner(ctx context.Context, modelName, skillsDir string) (travel.Planner, error) {
	if os.Getenv("GEMINI_API_KEY") == "" && os.Getenv("GOOGLE_API_KEY") == "" {
		return nil, errors.New("GEMINI_API_KEY または GOOGLE_API_KEY を設定してください")
	}
	g := genkit.Init(ctx,
		genkit.WithPlugins(&googlegenai.GoogleAI{}),
		genkit.WithDefaultModel("googleai/"+modelName),
	)
	return genkittravel.NewPlanner(genkittravel.DefineFlow(g, skillsDir)), nil
}

func modelOr(fallback string) string {
	if v := os.Getenv("GEMINI_MODEL"); v != "" {
		return v
	}
	return fallback
}
