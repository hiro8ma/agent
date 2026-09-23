// Package travelplanner は旅行プランナー。
//
// 3 つの調査を並列で走らせ、その結果を順次まとめる。
//
//	research_phase（並列）  観光スポット / レストラン / 交通手段
//	  ↓
//	schedule_planner（順次） 日程表を作る
//	  ↓
//	budget_reporter（順次）  予算を出す
//
// 主題は並列と順次の組み立てになる。
// どちらも実行順序をコードで固定する。LLM に順番を決めさせない。
package travelplanner

import (
	"fmt"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"

	"github.com/hiro8ma/agent/go/internal/travel"
)

// SearchSpots は観光スポットを検索する。
func SearchSpots(_ agent.Context, in travel.SearchSpotsInput) (travel.SearchSpotsOutput, error) {
	return travel.SearchSpots(in), nil
}

// SearchRestaurants はレストランを検索する。
func SearchRestaurants(_ agent.Context, in travel.SearchRestaurantsInput) (travel.SearchRestaurantsOutput, error) {
	return travel.SearchRestaurants(in), nil
}

// SearchTransport は交通手段を検索する。
func SearchTransport(_ agent.Context, in travel.SearchTransportInput) (travel.SearchTransportOutput, error) {
	return travel.SearchTransport(in), nil
}

// newTools はツールを組み立てる。
func newTools() (spots, restaurants, transport tool.Tool, err error) {
	spots, err = functiontool.New(functiontool.Config{
		Name:        "search_spots",
		Description: "旅行先の観光スポットを検索する。好みのジャンルで絞り込める。",
	}, SearchSpots)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("spots tool: %w", err)
	}
	restaurants, err = functiontool.New(functiontool.Config{
		Name:        "search_restaurants",
		Description: "旅行先のレストランを検索する。料理のジャンルで絞り込める。",
	}, SearchRestaurants)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("restaurants tool: %w", err)
	}
	transport, err = functiontool.New(functiontool.Config{
		Name:        "search_transport",
		Description: "出発地から目的地への交通手段を検索する。",
	}, SearchTransport)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("transport tool: %w", err)
	}
	return spots, restaurants, transport, nil
}
