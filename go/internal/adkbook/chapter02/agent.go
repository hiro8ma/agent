package chapter02

import (
	"context"
	"fmt"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagents/parallelagent"
	"google.golang.org/adk/v2/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/v2/model/gemini"
	"google.golang.org/adk/v2/tool"
)

// ModelName は教材のメインモデル。既定に依存しない。
const ModelName = "gemini-3.5-flash"

// 出力を次の段へ渡す鍵。
//
// 並列で走った 3 つの結果を、後段が読む必要がある。
// 鍵を通してセッションの状態に置くのが ADK のやり方になる。
// 後段のエージェントは前段の出力を直接受け取らず、状態から読む。
const (
	keySpots      = "spots_result"
	keyRestaurant = "restaurants_result"
	keyTransport  = "transport_result"
	keySchedule   = "schedule_result"
	keyBudget     = "budget_result"
)

// New は旅行プランナーを組み立てる。
//
// 構成は 2 段になる。
//
//	research_phase   並列。3 つの調査が同時に走る
//	schedule_planner 順次。調査結果を読んで日程表を作る
//	budget_reporter  順次。日程表を読んで予算を出す
//
// 並列にするのは 3 つの調査が互いに依存しないため。
// 順次にするのは日程表が調査結果を、予算が日程表を必要とするため。
// **依存関係がそのまま構成になる。**
func New(ctx context.Context, apiKey string) (agent.Agent, error) {
	m, err := gemini.NewModel(ctx, ModelName, &genai.ClientConfig{APIKey: apiKey})
	if err != nil {
		return nil, fmt.Errorf("create model: %w", err)
	}

	spotsTool, restaurantsTool, transportTool, err := newTools()
	if err != nil {
		return nil, err
	}

	// --- 調査フェーズ。3 つが並列に走る ---

	spotAgent, err := llmagent.New(llmagent.Config{
		Name:        "spot_researcher",
		Model:       m,
		Description: "観光スポットの専門家",
		Instruction: "あなたは観光スポットの専門家です。" +
			"ユーザーの旅行先と好みに基づいて、おすすめの観光スポットを検索してください。" +
			"検索結果をもとに、各スポットの特徴と所要時間を簡潔にまとめてください。",
		Tools:     []tool.Tool{spotsTool},
		OutputKey: keySpots,
	})
	if err != nil {
		return nil, fmt.Errorf("spot researcher: %w", err)
	}

	restaurantAgent, err := llmagent.New(llmagent.Config{
		Name:        "restaurant_researcher",
		Model:       m,
		Description: "グルメの専門家",
		Instruction: "あなたはグルメの専門家です。" +
			"ユーザーの旅行先と食の好みに基づいて、おすすめのレストランを検索してください。" +
			"検索結果をもとに、各レストランの特徴と価格帯を簡潔にまとめてください。",
		Tools:     []tool.Tool{restaurantsTool},
		OutputKey: keyRestaurant,
	})
	if err != nil {
		return nil, fmt.Errorf("restaurant researcher: %w", err)
	}

	transportAgent, err := llmagent.New(llmagent.Config{
		Name:        "transport_researcher",
		Model:       m,
		Description: "交通手段の専門家",
		Instruction: "あなたは交通手段の専門家です。" +
			"ユーザーの出発地と目的地に基づいて、利用可能な交通手段を検索してください。" +
			"検索結果をもとに、各手段の所要時間・料金・おすすめ度を簡潔にまとめてください。",
		Tools:     []tool.Tool{transportTool},
		OutputKey: keyTransport,
	})
	if err != nil {
		return nil, fmt.Errorf("transport researcher: %w", err)
	}

	researchPhase, err := parallelagent.New(parallelagent.Config{
		AgentConfig: agent.Config{
			Name:        "research_phase",
			Description: "観光・グルメ・交通の 3 つを並列に調べる",
			SubAgents:   []agent.Agent{spotAgent, restaurantAgent, transportAgent},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("research phase: %w", err)
	}

	// --- 計画フェーズ。調査結果を読んで順に組み立てる ---

	scheduleAgent, err := llmagent.New(llmagent.Config{
		Name:        "schedule_planner",
		Model:       m,
		Description: "旅行スケジュールの作成担当",
		Instruction: "あなたは旅行スケジュールの作成担当です。" +
			"これまでの調査結果（観光スポット・レストラン・交通手段）をもとに、" +
			"効率的な日程表を作成してください。\n" +
			"以下の形式で出力してください。\n" +
			"- 日ごとに時間帯を区切る（午前・昼・午後・夕方・夜）\n" +
			"- 各時間帯にスポットまたはレストランを配置\n" +
			"- 移動時間も考慮する",
		OutputKey: keySchedule,
	})
	if err != nil {
		return nil, fmt.Errorf("schedule planner: %w", err)
	}

	budgetAgent, err := llmagent.New(llmagent.Config{
		Name:        "budget_reporter",
		Model:       m,
		Description: "旅行予算の計算担当",
		Instruction: "あなたは旅行予算の計算担当です。" +
			"これまでの調査結果と作成された日程表をもとに、" +
			"旅行全体の概算予算を計算してください。\n" +
			"以下の項目ごとに金額を算出し、合計を出してください。\n" +
			"- 交通費（往復 + 現地移動）\n" +
			"- 食費（朝食・昼食・夕食 × 日数）\n" +
			"- 入場料・アクティビティ費\n" +
			"- 合計（税・チップ込みの概算）",
		OutputKey: keyBudget,
	})
	if err != nil {
		return nil, fmt.Errorf("budget reporter: %w", err)
	}

	root, err := sequentialagent.New(sequentialagent.Config{
		AgentConfig: agent.Config{
			Name:        "travel_planner",
			Description: "旅行プランナー。調査 → 日程作成 → 予算計算を順次実行する",
			SubAgents:   []agent.Agent{researchPhase, scheduleAgent, budgetAgent},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("travel planner: %w", err)
	}
	return root, nil
}
