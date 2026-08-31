// Package chapter02 は教材の第 2 章、旅行プランナー。
//
// 3 つの調査を並列で走らせ、その結果を順次まとめる。
//
//	research_phase（並列）  観光スポット / レストラン / 交通手段
//	  ↓
//	schedule_planner（順次） 日程表を作る
//	  ↓
//	budget_reporter（順次）  予算を出す
//
// 並列と順次を組み合わせるのが第 2 章の主題になる。
// どちらも実行順序をコードで固定する。LLM に順番を決めさせない。
package chapter02

import (
	"fmt"
	"strings"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// Spot は観光スポット 1 件。
type Spot struct {
	Name          string  `json:"name"`
	Genre         string  `json:"genre"`
	Rating        float64 `json:"rating"`
	DurationHours float64 `json:"durationHours"`
}

// Restaurant はレストラン 1 件。
type Restaurant struct {
	Name       string  `json:"name"`
	Cuisine    string  `json:"cuisine"`
	Rating     float64 `json:"rating"`
	PriceRange string  `json:"priceRange"`
	BudgetYen  int     `json:"budgetYen"`
}

// Transport は交通手段 1 件。
type Transport struct {
	Mode        string `json:"mode"`
	DurationMin int    `json:"durationMin"`
	FareYen     int    `json:"fareYen"`
	Note        string `json:"note"`
}

// 本番では地図や予約の API を呼ぶ。教材と同じく手元の固定データで代える。
// 第 2 章の主題は並列と順次の組み立てなので、
// ツールの中身は結果の形だけ合っていればよい。
var (
	spotsDB = map[string][]Spot{
		"京都": {
			{"金閣寺", "歴史", 4.5, 1.5},
			{"嵐山竹林", "自然", 4.7, 2.0},
			{"伏見稲荷大社", "歴史", 4.8, 2.0},
			{"京都国立博物館", "アート", 4.3, 2.5},
		},
		"沖縄": {
			{"美ら海水族館", "自然", 4.6, 3.0},
			{"首里城", "歴史", 4.4, 1.5},
			{"古宇利島", "自然", 4.7, 3.0},
			{"国際通り", "ショッピング", 4.2, 2.0},
		},
		"東京": {
			{"浅草寺", "歴史", 4.5, 1.5},
			{"チームラボボーダレス", "アート", 4.6, 2.5},
			{"明治神宮", "歴史", 4.4, 1.5},
			{"新宿御苑", "自然", 4.3, 2.0},
		},
	}

	restaurantsDB = map[string][]Restaurant{
		"京都": {
			{"祇園おかる", "和食", 4.4, "1000-2000 円", 1500},
			{"京都吉兆", "懐石", 4.9, "20000 円以上", 25000},
			{"一乗寺中谷", "甘味", 4.3, "500-1500 円", 1000},
		},
		"沖縄": {
			{"花笠食堂", "沖縄料理", 4.2, "800-1500 円", 1200},
			{"あぐー豚しゃぶ", "和食", 4.6, "3000-5000 円", 4000},
			{"ブルーシール", "スイーツ", 4.1, "500 円前後", 500},
		},
		"東京": {
			{"すきやばし次郎", "寿司", 4.8, "30000 円以上", 35000},
			{"天丼てんや", "和食", 4.0, "500-1000 円", 800},
			{"ラーメン二郎", "ラーメン", 4.2, "800-1200 円", 1000},
		},
	}

	transportDB = map[string]map[string][]Transport{
		"東京": {
			"京都": {
				{"新幹線のぞみ", 140, 14000, "最速。本数が多い"},
				{"高速バス", 480, 5000, "夜行なら宿泊費が浮く"},
				{"飛行機", 90, 12000, "空港までの移動が別に要る"},
			},
			"沖縄": {
				{"飛行機", 165, 25000, "実質これ 1 択"},
			},
		},
	}
)

// SearchSpotsInput は観光スポット検索の入力。
type SearchSpotsInput struct {
	// Destination は旅行先の都市名（例 京都）。
	Destination string `json:"destination"`
	// Preferences は好みのジャンル（例 歴史、自然、アート）。
	Preferences string `json:"preferences"`
}

// SearchSpotsOutput は観光スポット検索の結果。
//
// 失敗も構造化して返す。エラーを返すとモデルが理由を読めない。
type SearchSpotsOutput struct {
	Destination string `json:"destination"`
	Preferences string `json:"preferences,omitempty"`
	Spots       []Spot `json:"spots,omitempty"`
	Error       string `json:"error,omitempty"`
}

// SearchSpots は観光スポットを検索する。
func SearchSpots(_ agent.Context, in SearchSpotsInput) (SearchSpotsOutput, error) {
	spots, ok := spotsDB[strings.TrimSpace(in.Destination)]
	if !ok {
		return SearchSpotsOutput{
			Destination: in.Destination,
			Error:       fmt.Sprintf("%s の観光スポットデータは見つからなかった", in.Destination),
		}, nil
	}

	// 好みで絞る。ただし 0 件になるなら絞らない。
	// 絞った結果が空だと、モデルは「データが無い」と読んでしまう。
	if in.Preferences != "" {
		var filtered []Spot
		for _, s := range spots {
			if strings.Contains(s.Genre, in.Preferences) {
				filtered = append(filtered, s)
			}
		}
		if len(filtered) > 0 {
			spots = filtered
		}
	}
	return SearchSpotsOutput{
		Destination: in.Destination, Preferences: in.Preferences, Spots: spots,
	}, nil
}

// SearchRestaurantsInput はレストラン検索の入力。
type SearchRestaurantsInput struct {
	Destination string `json:"destination"`
	// Cuisine は料理のジャンル（例 和食、寿司）。
	Cuisine string `json:"cuisine"`
}

// SearchRestaurantsOutput はレストラン検索の結果。
type SearchRestaurantsOutput struct {
	Destination string       `json:"destination"`
	Cuisine     string       `json:"cuisine,omitempty"`
	Restaurants []Restaurant `json:"restaurants,omitempty"`
	Error       string       `json:"error,omitempty"`
}

// SearchRestaurants はレストランを検索する。
func SearchRestaurants(_ agent.Context, in SearchRestaurantsInput) (SearchRestaurantsOutput, error) {
	list, ok := restaurantsDB[strings.TrimSpace(in.Destination)]
	if !ok {
		return SearchRestaurantsOutput{
			Destination: in.Destination,
			Error:       fmt.Sprintf("%s のレストランデータは見つからなかった", in.Destination),
		}, nil
	}
	if in.Cuisine != "" {
		var filtered []Restaurant
		for _, r := range list {
			if strings.Contains(r.Cuisine, in.Cuisine) {
				filtered = append(filtered, r)
			}
		}
		if len(filtered) > 0 {
			list = filtered
		}
	}
	return SearchRestaurantsOutput{
		Destination: in.Destination, Cuisine: in.Cuisine, Restaurants: list,
	}, nil
}

// SearchTransportInput は交通手段検索の入力。
type SearchTransportInput struct {
	Origin      string `json:"origin"`
	Destination string `json:"destination"`
}

// SearchTransportOutput は交通手段検索の結果。
type SearchTransportOutput struct {
	Origin      string      `json:"origin"`
	Destination string      `json:"destination"`
	Options     []Transport `json:"options,omitempty"`
	Error       string      `json:"error,omitempty"`
}

// SearchTransport は交通手段を検索する。
func SearchTransport(_ agent.Context, in SearchTransportInput) (SearchTransportOutput, error) {
	from, ok := transportDB[strings.TrimSpace(in.Origin)]
	if !ok {
		return SearchTransportOutput{
			Origin: in.Origin, Destination: in.Destination,
			Error: fmt.Sprintf("%s 発の交通データは見つからなかった", in.Origin),
		}, nil
	}
	opts, ok := from[strings.TrimSpace(in.Destination)]
	if !ok {
		return SearchTransportOutput{
			Origin: in.Origin, Destination: in.Destination,
			Error: fmt.Sprintf("%s から %s への交通データは見つからなかった", in.Origin, in.Destination),
		}, nil
	}
	return SearchTransportOutput{
		Origin: in.Origin, Destination: in.Destination, Options: opts,
	}, nil
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
