// Package travelplanner は旅行プランナーの Genkit 版。
//
// ADK 版（internal/adk/travelplanner）と同じ構成を、Genkit のフロー 1 つで組む。
//
//	調査（並列）  観光スポット / レストラン / 交通手段
//	  ↓
//	日程（順次）  調査結果をプロンプトに埋めて日程表を作る
//	  ↓
//	予算（順次）  日程表と調査結果から予算を構造化出力で受け取る
//
// Genkit の安定版には並列と順次の部品が無い。
// 並列は goroutine、順次は関数呼び出しの並びで書く。
package travelplanner

import (
	"context"
	"fmt"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/genkit"
	"golang.org/x/sync/errgroup"

	"github.com/hiro8ma/agent/go/internal/travel"
)

// ModelName は ADK 版と同じモデル。googleai プラグインでは接頭辞 googleai/ を付けて引く。
const ModelName = "gemini-3.8-flash"

// FlowName は Developer UI と HTTP の口に出るフローの名前。
const FlowName = "travelPlanner"

const (
	researchMaxTurns = 3
	planMaxTurns     = 1
)

const (
	spotSystem = "あなたは観光スポットの専門家です。" +
		"ユーザーの旅行先と好みに基づいて、おすすめの観光スポットを検索してください。" +
		"検索結果をもとに、各スポットの特徴と所要時間を簡潔にまとめてください。"
	restaurantSystem = "あなたはグルメの専門家です。" +
		"ユーザーの旅行先と食の好みに基づいて、おすすめのレストランを検索してください。" +
		"検索結果をもとに、各レストランの特徴と価格帯を簡潔にまとめてください。"
	transportSystem = "あなたは交通手段の専門家です。" +
		"ユーザーの出発地と目的地に基づいて、利用可能な交通手段を検索してください。" +
		"検索結果をもとに、各手段の所要時間・料金・おすすめ度を簡潔にまとめてください。"
	scheduleSystem = "あなたは旅行スケジュールの作成担当です。"
	budgetSystem   = "あなたは旅行予算の計算担当です。"
)

const researchPrompt = "旅行の依頼: %s"

const schedulePrompt = "旅行の依頼: %s\n" +
	"次の調査結果をもとに、効率的な日程表を作成してください。\n" +
	"観光スポット: %s\n" +
	"レストラン: %s\n" +
	"交通手段: %s\n" +
	"以下の形式で出力してください。\n" +
	"- 日ごとに時間帯を区切る（午前・昼・午後・夕方・夜）\n" +
	"- 各時間帯にスポットまたはレストランを配置\n" +
	"- 移動時間も考慮する"

const budgetPrompt = "次の日程表と調査結果をもとに、旅行全体の概算予算を計算してください。\n" +
	"日程表: %s\n" +
	"交通手段: %s\n" +
	"レストラン: %s\n" +
	"観光スポット: %s\n" +
	"以下の項目ごとに金額を算出し、合計を出してください。\n" +
	"- 交通費（往復 + 現地移動）\n" +
	"- 食費（朝食・昼食・夕食 × 日数）\n" +
	"- 入場料・アクティビティ費\n" +
	"- 合計（税・チップ込みの概算）"

// TripRequest はフローの入力。
type TripRequest struct {
	Request string `json:"request" jsonschema_description:"旅行の依頼（例 東京から京都に2泊3日で行きたい。歴史が好き）"`
}

// Research は並列に走った 3 つの調査の結果。
type Research struct {
	Spots       string `json:"spots"`
	Restaurants string `json:"restaurants"`
	Transport   string `json:"transport"`
}

// Budget は旅行全体の概算予算。ADK 版の出力スキーマと同じ形にする。
type Budget struct {
	Currency          string   `json:"currency" jsonschema:"enum=JPY"`
	TransportationYen int      `json:"transportation_yen"`
	FoodYen           int      `json:"food_yen"`
	ActivitiesYen     int      `json:"activities_yen"`
	TotalYen          int      `json:"total_yen"`
	Assumptions       []string `json:"assumptions"`
}

// TripPlan はフローの出力。
type TripPlan struct {
	Research Research `json:"research"`
	Schedule string   `json:"schedule"`
	Budget   Budget   `json:"budget"`
}

type researchers struct {
	spots, restaurants, transport ai.ToolRef
}

// DefineFlow は検索ツールと旅行プランナーのフローを g に登録する。モデルは g の既定モデルを使う。
func DefineFlow(g *genkit.Genkit) *core.Flow[TripRequest, TripPlan, struct{}] {
	tools := researchers{
		spots: genkit.DefineTool(g, "search_spots", "旅行先の観光スポットを検索する。好みのジャンルで絞り込める。",
			func(_ *ai.ToolContext, in travel.SearchSpotsInput) (travel.SearchSpotsOutput, error) {
				return travel.SearchSpots(in), nil
			}),
		restaurants: genkit.DefineTool(g, "search_restaurants", "旅行先のレストランを検索する。料理のジャンルで絞り込める。",
			func(_ *ai.ToolContext, in travel.SearchRestaurantsInput) (travel.SearchRestaurantsOutput, error) {
				return travel.SearchRestaurants(in), nil
			}),
		transport: genkit.DefineTool(g, "search_transport", "出発地から目的地への交通手段を検索する。",
			func(_ *ai.ToolContext, in travel.SearchTransportInput) (travel.SearchTransportOutput, error) {
				return travel.SearchTransport(in), nil
			}),
	}

	return genkit.DefineFlow(g, FlowName, func(ctx context.Context, req TripRequest) (TripPlan, error) {
		research, err := runResearch(ctx, g, tools, req.Request)
		if err != nil {
			return TripPlan{}, err
		}

		schedule, err := genkit.GenerateText(ctx, g,
			ai.WithSystem(scheduleSystem),
			ai.WithPrompt(schedulePrompt, req.Request, research.Spots, research.Restaurants, research.Transport),
			ai.WithMaxTurns(planMaxTurns),
		)
		if err != nil {
			return TripPlan{}, fmt.Errorf("schedule: %w", err)
		}

		budget, _, err := genkit.GenerateData[Budget](ctx, g,
			ai.WithSystem(budgetSystem),
			ai.WithPrompt(budgetPrompt, schedule, research.Transport, research.Restaurants, research.Spots),
			ai.WithMaxTurns(planMaxTurns),
		)
		if err != nil {
			return TripPlan{}, fmt.Errorf("budget: %w", err)
		}
		return TripPlan{Research: research, Schedule: schedule, Budget: *budget}, nil
	})
}

func runResearch(ctx context.Context, g *genkit.Genkit, tools researchers, request string) (Research, error) {
	var r Research
	eg, ctx := errgroup.WithContext(ctx)
	eg.Go(func() (err error) {
		r.Spots, err = investigate(ctx, g, spotSystem, tools.spots, request)
		return err
	})
	eg.Go(func() (err error) {
		r.Restaurants, err = investigate(ctx, g, restaurantSystem, tools.restaurants, request)
		return err
	})
	eg.Go(func() (err error) {
		r.Transport, err = investigate(ctx, g, transportSystem, tools.transport, request)
		return err
	})
	if err := eg.Wait(); err != nil {
		return Research{}, err
	}
	return r, nil
}

func investigate(ctx context.Context, g *genkit.Genkit, system string, tool ai.ToolRef, request string) (string, error) {
	text, err := genkit.GenerateText(ctx, g,
		ai.WithSystem(system),
		ai.WithPrompt(researchPrompt, request),
		ai.WithTools(tool),
		ai.WithMaxTurns(researchMaxTurns),
	)
	if err != nil {
		return "", fmt.Errorf("%s: %w", tool.Name(), err)
	}
	if strings.TrimSpace(text) == "" {
		return "", fmt.Errorf("%s: 調査結果が空", tool.Name())
	}
	return text, nil
}
