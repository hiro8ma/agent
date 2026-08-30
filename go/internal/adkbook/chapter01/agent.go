// Package chapter01 は教材の第 1 章、天気取得エージェントの最小構成。
//
// Python 版は `root_agent` という変数名の規約でエントリーポイントを決め、
// ツールのスキーマを型注釈と docstring から自動で組み立てる。
// Go 版は両方を明示的に書く。書く量は増えるが、
// 繋ぎ忘れがコンパイルで出る。
package chapter01

import (
	"context"
	"fmt"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model/gemini"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// ModelName は教材のメインモデル。
//
// ADK v2.2.0 の LlmAgent の既定は gemini-3-flash-preview になるが、
// 暗黙の既定に依存しない。既定はバージョンで変わり、
// 変わったことに気づけないまま結果だけが動く。
const ModelName = "gemini-3.5-flash"

// weather は手元の固定データ。第 1 章は外部 API を呼ばず、
// ツールが呼ばれる仕組みだけを見る。
var weather = map[string]string{
	"tokyo":   "晴れ、気温 28 度、湿度 55%",
	"osaka":   "曇り、気温 30 度、湿度 70%",
	"sapporo": "雨、気温 21 度、湿度 85%",
}

// WeatherInput はツールの入力。
//
// json タグがそのままモデルへ渡るスキーマになる。
// Python 版が docstring から組み立てるところを、Go は型で書く。
type WeatherInput struct {
	// City は都市名。英語の小文字で渡す（例 tokyo）。
	City string `json:"city"`
}

// WeatherOutput はツールの出力。
//
// 失敗も構造化して返す。エラーを返すとモデルが理由を読めず、
// 「登録されていない都市」と「呼び出しに失敗した」を区別できない。
type WeatherOutput struct {
	Status       string `json:"status"`
	Report       string `json:"report,omitempty"`
	ErrorMessage string `json:"errorMessage,omitempty"`
}

// GetWeather は指定した都市の現在の天気を返す。
func GetWeather(_ agent.Context, in WeatherInput) (WeatherOutput, error) {
	key := strings.ToLower(strings.TrimSpace(in.City))
	report, ok := weather[key]
	if !ok {
		return WeatherOutput{
			Status:       "error",
			ErrorMessage: fmt.Sprintf("%s の天気は登録されていない", in.City),
		}, nil
	}
	return WeatherOutput{Status: "success", Report: report}, nil
}

const instruction = "あなたは天気を答えるエージェントです。" +
	"都市の天気を聞かれたら get_weather を呼び、その結果だけを使って答えます。" +
	"天気と無関係な質問には答えません。" +
	"ツールが error を返したら、登録されていない都市であることを伝えます。"

// New は天気エージェントを組み立てる。
//
// Python 版の root_agent にあたるが、変数名の規約ではなく
// 呼び出し側が受け取って launcher に渡す。
func New(ctx context.Context, apiKey string) (agent.Agent, error) {
	model, err := gemini.NewModel(ctx, ModelName, &genai.ClientConfig{APIKey: apiKey})
	if err != nil {
		return nil, fmt.Errorf("create model: %w", err)
	}

	weatherTool, err := functiontool.New(functiontool.Config{
		Name:        "get_weather",
		Description: "指定した都市の現在の天気を返す。都市名は英語の小文字で渡す。",
	}, GetWeather)
	if err != nil {
		return nil, fmt.Errorf("create tool: %w", err)
	}

	a, err := llmagent.New(llmagent.Config{
		Name:        "weather_agent",
		Model:       model,
		Description: "都市の天気を答えるエージェント",
		Instruction: instruction,
		Tools:       []tool.Tool{weatherTool},
	})
	if err != nil {
		return nil, fmt.Errorf("create agent: %w", err)
	}
	return a, nil
}
