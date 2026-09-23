package travel

import (
	"context"
	"iter"
)

// Planner は旅行プランを作る。ADK 版と Genkit 版が満たす。
type Planner interface {
	Plan(ctx context.Context, req *PlanRequest) iter.Seq2[*PlanResponse, error]
}

// PlanRequest は旅行プランの依頼。
type PlanRequest struct {
	UserID    string `json:"userId,omitempty"`
	SessionID string `json:"sessionId"`
	// Message は利用者の依頼文（例 東京から京都に2泊3日で行きたい。歴史が好き）。
	Message string `json:"message"`
}

// PlanResponse は Delta と Plan のどちらか一方を持つ。Plan を持つ応答が列の最後になる。
type PlanResponse struct {
	// Delta は生成中の日程表の差分。
	Delta string `json:"delta,omitempty"`
	Plan  *Plan  `json:"plan,omitempty"`
}

// Plan は完成した旅行プラン。
type Plan struct {
	Schedule string `json:"schedule"`
	Budget   Budget `json:"budget"`
}

// Budget は旅行全体の概算予算。ADK 版の出力スキーマと Genkit 版の構造化出力が同じ形で返す。
type Budget struct {
	Currency          string   `json:"currency" jsonschema:"enum=JPY"`
	TransportationYen int      `json:"transportation_yen"`
	FoodYen           int      `json:"food_yen"`
	ActivitiesYen     int      `json:"activities_yen"`
	TotalYen          int      `json:"total_yen"`
	Assumptions       []string `json:"assumptions"`
}
