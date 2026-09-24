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
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/genai"

	"github.com/hiro8ma/agent/go/internal/adk/callguard"
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

	modelName := travelplanner.ModelName
	if v := os.Getenv("GEMINI_MODEL"); v != "" {
		modelName = v
	}
	m, err := gemini.NewModel(ctx, modelName, &genai.ClientConfig{APIKey: apiKey})
	if err != nil {
		log.Fatalf("failed to create model: %v", err)
	}
	var skills tool.Toolset
	if dir := os.Getenv("TRAVEL_SKILLS_DIR"); dir != "" {
		skills, err = travelplanner.NewSkillToolset(ctx, dir)
		if err != nil {
			log.Fatalf("failed to load skills: %v", err)
		}
	}
	// 無料枠は 1 分あたり 5 回で、調査の 3 並列だけで越える。429 の待ち時間に従って再試行する
	a, err := travelplanner.NewWithModel(llmretry.Wrap(m, llmretry.DefaultPolicy()), skills)
	if err != nil {
		log.Fatalf("failed to build agent: %v", err)
	}

	// 通常の 1 回は調査 3 つが各 2 回、日程と予算が各 1 回で、モデルを 8 回呼ぶ
	guard, err := callguard.New(callguard.Limits{MaxLLMCalls: 20, MaxSameToolCalls: 3})
	if err != nil {
		log.Fatalf("failed to build callguard: %v", err)
	}

	l := full.NewLauncher()
	cfg := &launcher.Config{
		AgentLoader:  agent.NewSingleLoader(a),
		PluginConfig: runner.PluginConfig{Plugins: []*plugin.Plugin{guard}},
	}
	if err := l.Execute(ctx, cfg, os.Args[1:]); err != nil {
		log.Fatalf("run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}
