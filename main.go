package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
)

// CalcInput は計算ツールの入力スキーマ
type CalcInput struct {
	A  float64 `json:"a" jsonschema:"description=最初の数値"`
	B  float64 `json:"b" jsonschema:"description=2番目の数値"`
	Op string  `json:"op" jsonschema:"description=演算子: add, sub, mul, div"`
}

// CalcOutput は計算ツールの出力スキーマ
type CalcOutput struct {
	Result float64 `json:"result"`
}

func main() {
	ctx := context.Background()

	// APIキーの確認
	if os.Getenv("GEMINI_API_KEY") == "" {
		log.Fatal("GEMINI_API_KEY environment variable is not set")
	}

	// Genkitの初期化（Google AIプラグインを使用）
	g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}))

	// 使用するモデルを定義
	model := googlegenai.GoogleAIModel(g, "gemini-2.0-flash")

	// === デモ1: シンプルなテキスト生成 ===
	fmt.Println("=== デモ1: シンプルなテキスト生成 ===")
	response, err := genkit.GenerateText(ctx, g,
		ai.WithModel(model),
		ai.WithPrompt("Goプログラミング言語の良いところを3つ、簡潔に教えてください。"),
	)
	if err != nil {
		log.Fatalf("Failed to generate text: %v", err)
	}
	fmt.Println(response)
	fmt.Println()

	// === デモ2: Tool Calling（計算エージェント）===
	fmt.Println("=== デモ2: Tool Calling（計算エージェント）===")

	// 計算ツールを定義
	calcTool := genkit.DefineTool(g, "calculator",
		"四則演算を行う計算ツール。add(足し算), sub(引き算), mul(掛け算), div(割り算)をサポート",
		func(ctx *ai.ToolContext, input CalcInput) (CalcOutput, error) {
			var result float64
			switch input.Op {
			case "add":
				result = input.A + input.B
			case "sub":
				result = input.A - input.B
			case "mul":
				result = input.A * input.B
			case "div":
				if input.B == 0 {
					return CalcOutput{}, fmt.Errorf("0で割ることはできません")
				}
				result = input.A / input.B
			default:
				return CalcOutput{}, fmt.Errorf("不明な演算子: %s", input.Op)
			}
			fmt.Printf("  [Tool] %v %s %v = %v\n", input.A, input.Op, input.B, result)
			return CalcOutput{Result: result}, nil
		},
	)

	// ツールを使った生成
	calcResponse, err := genkit.Generate(ctx, g,
		ai.WithModel(model),
		ai.WithPrompt("123と456を足して、その結果を2倍してください。計算にはcalculatorツールを使ってください。"),
		ai.WithTools(calcTool),
	)
	if err != nil {
		log.Fatalf("Failed to generate with tools: %v", err)
	}
	fmt.Println("回答:", calcResponse.Text())
	fmt.Println()

	fmt.Println("=== 完了 ===")
}
