package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"github.com/firebase/genkit/go/plugins/ollama"
)

// ========================================
// 型定義
// ========================================

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

// JokeInput はジョークFlowの入力
type JokeInput struct {
	Topic string `json:"topic"`
}

// JokeOutput はジョークFlowの出力（構造化出力）
type JokeOutput struct {
	Setup     string `json:"setup" jsonschema:"description=ジョークの前振り"`
	Punchline string `json:"punchline" jsonschema:"description=ジョークのオチ"`
	Rating    int    `json:"rating" jsonschema:"description=面白さ評価 1-10"`
}

// SentimentOutput は感情分析の構造化出力
type SentimentOutput struct {
	Sentiment  string   `json:"sentiment" jsonschema:"description=感情: positive, negative, neutral"`
	Confidence float64  `json:"confidence" jsonschema:"description=信頼度 0.0-1.0"`
	Keywords   []string `json:"keywords" jsonschema:"description=重要なキーワード"`
}

// グローバル変数（サーバーモードで使用）
var (
	genkitInstance *genkit.Genkit
	defaultModel   ai.Model
	calcTool       ai.Tool
)

func main() {
	ctx := context.Background()

	// 動作モードの確認
	mode := os.Getenv("MODE")
	if mode == "" {
		mode = "demo" // デフォルトはデモモード
	}

	// APIキーの確認（Gemini使用時）
	useGemini := os.Getenv("GEMINI_API_KEY") != ""
	useOllama := os.Getenv("USE_OLLAMA") == "true"

	if !useGemini && !useOllama {
		log.Fatal("GEMINI_API_KEY または USE_OLLAMA=true を設定してください")
	}

	// Genkitの初期化（プラグインを動的に選択）
	if useGemini && useOllama {
		ollamaAddr := os.Getenv("OLLAMA_HOST")
		if ollamaAddr == "" {
			ollamaAddr = "http://localhost:11434"
		}
		genkitInstance = genkit.Init(ctx, genkit.WithPlugins(
			&googlegenai.GoogleAI{},
			&ollama.Ollama{ServerAddress: ollamaAddr},
		))
	} else if useGemini {
		genkitInstance = genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}))
	} else {
		ollamaAddr := os.Getenv("OLLAMA_HOST")
		if ollamaAddr == "" {
			ollamaAddr = "http://localhost:11434"
		}
		genkitInstance = genkit.Init(ctx, genkit.WithPlugins(&ollama.Ollama{ServerAddress: ollamaAddr}))
	}

	// 使用するモデルを選択
	if useGemini {
		defaultModel = googlegenai.GoogleAIModel(genkitInstance, "gemini-2.0-flash")
		fmt.Println("Using: Google AI (Gemini)")
	} else {
		defaultModel = ollama.Model(genkitInstance, "llama3.2")
		fmt.Println("Using: Ollama (llama3.2)")
	}

	// ========================================
	// ツール定義
	// ========================================
	calcTool = genkit.DefineTool(genkitInstance, "calculator",
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

	// ========================================
	// Flow定義（再利用可能なワークフロー）
	// ========================================

	// ジョークを生成するFlow（構造化出力付き）
	jokeFlow := genkit.DefineFlow(genkitInstance, "tellJoke",
		func(ctx context.Context, input JokeInput) (JokeOutput, error) {
			result, _, err := genkit.GenerateData[JokeOutput](ctx, genkitInstance,
				ai.WithModel(defaultModel),
				ai.WithPrompt(fmt.Sprintf(`
あなたはコメディアンです。「%s」についてのジョークを考えてください。
必ずJSON形式で以下のフィールドを含めてください：
- setup: ジョークの前振り
- punchline: ジョークのオチ
- rating: 面白さの自己評価（1-10）
`, input.Topic)),
			)
			if err != nil {
				return JokeOutput{}, err
			}
			return *result, nil
		},
	)

	// 感情分析Flow（構造化出力のデモ）
	sentimentFlow := genkit.DefineFlow(genkitInstance, "analyzeSentiment",
		func(ctx context.Context, text string) (SentimentOutput, error) {
			result, _, err := genkit.GenerateData[SentimentOutput](ctx, genkitInstance,
				ai.WithModel(defaultModel),
				ai.WithPrompt(fmt.Sprintf(`
以下のテキストの感情を分析してください：
"%s"

JSON形式で回答してください。
`, text)),
			)
			if err != nil {
				return SentimentOutput{}, err
			}
			return *result, nil
		},
	)

	// ========================================
	// モード別実行
	// ========================================
	switch mode {
	case "server":
		runServer()
	default:
		runDemo(ctx)
	}

	// Flow変数を使用済みとしてマーク（未使用エラー回避）
	_ = jokeFlow
	_ = sentimentFlow
}

// runDemo はデモモードを実行
func runDemo(ctx context.Context) {
	// === デモ1: シンプルなテキスト生成 ===
	fmt.Println("\n=== デモ1: シンプルなテキスト生成 ===")
	response, err := genkit.GenerateText(ctx, genkitInstance,
		ai.WithModel(defaultModel),
		ai.WithPrompt("Goプログラミング言語の良いところを3つ、簡潔に教えてください。"),
	)
	if err != nil {
		log.Printf("Failed to generate text: %v", err)
	} else {
		fmt.Println(response)
	}

	// === デモ2: Tool Calling（計算エージェント）===
	fmt.Println("\n=== デモ2: Tool Calling（計算エージェント）===")
	calcResponse, err := genkit.Generate(ctx, genkitInstance,
		ai.WithModel(defaultModel),
		ai.WithPrompt("123と456を足して、その結果を2倍してください。計算にはcalculatorツールを使ってください。"),
		ai.WithTools(calcTool),
	)
	if err != nil {
		log.Printf("Failed to generate with tools: %v", err)
	} else {
		fmt.Println("回答:", calcResponse.Text())
	}

	// === デモ3: 構造化出力（感情分析）===
	fmt.Println("\n=== デモ3: 構造化出力（感情分析）===")
	sentimentResult, _, err := genkit.GenerateData[SentimentOutput](ctx, genkitInstance,
		ai.WithModel(defaultModel),
		ai.WithPrompt(`
以下のテキストの感情を分析してください：
"今日はとても素晴らしい天気で、気分が良いです！"

JSON形式で回答してください。
`),
	)
	if err != nil {
		log.Printf("Failed sentiment analysis: %v", err)
	} else {
		fmt.Printf("感情: %s (信頼度: %.2f)\n", sentimentResult.Sentiment, sentimentResult.Confidence)
		fmt.Printf("キーワード: %v\n", sentimentResult.Keywords)
	}

	// === デモ4: 構造化出力（ジョーク生成）===
	fmt.Println("\n=== デモ4: 構造化出力（ジョーク生成）===")
	jokeResult, _, err := genkit.GenerateData[JokeOutput](ctx, genkitInstance,
		ai.WithModel(defaultModel),
		ai.WithPrompt(`
あなたはコメディアンです。「プログラミング」についてのジョークを考えてください。
必ずJSON形式で以下のフィールドを含めてください：
- setup: ジョークの前振り
- punchline: ジョークのオチ
- rating: 面白さの自己評価（1-10）
`),
	)
	if err != nil {
		log.Printf("Failed to generate joke: %v", err)
	} else {
		fmt.Printf("前振り: %s\n", jokeResult.Setup)
		fmt.Printf("オチ: %s\n", jokeResult.Punchline)
		fmt.Printf("自己評価: %d/10\n", jokeResult.Rating)
	}

	fmt.Println("\n=== 完了 ===")
	fmt.Println("\nREST APIモードで起動するには: MODE=server go run main.go")
}

// runServer はREST APIサーバーを起動
func runServer() {
	mux := http.NewServeMux()

	// すべてのFlowをHandlerでエクスポート
	for _, flow := range genkit.ListFlows(genkitInstance) {
		name := flow.Desc().Name
		mux.HandleFunc("POST /api/"+name, genkit.Handler(flow))
		fmt.Printf("  Registered: POST /api/%s\n", name)
	}

	// シンプルなチャットエンドポイント（手動実装）
	mux.HandleFunc("POST /api/chat", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Message string `json:"message"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid JSON", http.StatusBadRequest)
			return
		}

		ctx := r.Context()
		response, err := genkit.Generate(ctx, genkitInstance,
			ai.WithModel(defaultModel),
			ai.WithPrompt(req.Message),
			ai.WithTools(calcTool),
		)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{
			"response": response.Text(),
		})
	})

	// ヘルスチェック
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("OK"))
	})

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	fmt.Printf("\n=== AI Agent REST API Server ===\n")
	fmt.Printf("Server starting on http://localhost:%s\n\n", port)
	fmt.Println("Endpoints:")
	fmt.Println("  POST /api/tellJoke       - ジョーク生成 {\"data\": {\"topic\": \"テーマ\"}}")
	fmt.Println("  POST /api/analyzeSentiment - 感情分析 {\"data\": \"テキスト\"}")
	fmt.Println("  POST /api/chat           - チャット {\"message\": \"メッセージ\"}")
	fmt.Println("  GET  /health             - ヘルスチェック")
	fmt.Println()

	if err := http.ListenAndServe(":"+port, mux); err != nil {
		log.Fatal(err)
	}
}
