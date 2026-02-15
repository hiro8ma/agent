package genkit

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/firebase/genkit/go/ai"
	genkitsdk "github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"github.com/firebase/genkit/go/plugins/ollama"
)

// App はGenkit AIエージェントアプリケーション
type App struct {
	g         *genkitsdk.Genkit
	model     ai.Model
	calcTool  ai.Tool
	store     *VectorStore
	embedder  ai.Embedder
	useGemini bool
}

// New はGenkit Appを初期化して返す
func New(ctx context.Context) *App {
	a := &App{
		store: NewVectorStore(),
	}

	// APIキーの確認（Gemini使用時）
	a.useGemini = os.Getenv("GEMINI_API_KEY") != ""
	useOllama := os.Getenv("USE_OLLAMA") == "true"

	if !a.useGemini && !useOllama {
		log.Fatal("GEMINI_API_KEY または USE_OLLAMA=true を設定してください")
	}

	// Genkitの初期化（プラグインを動的に選択）
	if a.useGemini && useOllama {
		ollamaAddr := os.Getenv("OLLAMA_HOST")
		if ollamaAddr == "" {
			ollamaAddr = "http://localhost:11434"
		}
		a.g = genkitsdk.Init(ctx, genkitsdk.WithPlugins(
			&googlegenai.GoogleAI{},
			&ollama.Ollama{ServerAddress: ollamaAddr},
		))
	} else if a.useGemini {
		a.g = genkitsdk.Init(ctx, genkitsdk.WithPlugins(&googlegenai.GoogleAI{}))
	} else {
		ollamaAddr := os.Getenv("OLLAMA_HOST")
		if ollamaAddr == "" {
			ollamaAddr = "http://localhost:11434"
		}
		a.g = genkitsdk.Init(ctx, genkitsdk.WithPlugins(&ollama.Ollama{ServerAddress: ollamaAddr}))
	}

	// 使用するモデルを選択
	if a.useGemini {
		a.model = googlegenai.GoogleAIModel(a.g, "gemini-2.0-flash")
		a.embedder = googlegenai.GoogleAIEmbedder(a.g, "text-embedding-004")
		fmt.Println("Using: Google AI (Gemini) with Embedder")
	} else {
		a.model = ollama.Model(a.g, "llama3.2")
		fmt.Println("Using: Ollama (llama3.2) - RAG disabled (no embedder)")
	}

	a.defineTools()
	a.defineFlows()

	return a
}

// Run は指定されたモードでアプリケーションを実行
func (a *App) Run(ctx context.Context) {
	mode := os.Getenv("MODE")
	if mode == "" {
		mode = "demo"
	}

	switch mode {
	case "server":
		a.RunServer()
	default:
		a.RunDemo(ctx)
	}
}

func (a *App) defineTools() {
	a.calcTool = genkitsdk.DefineTool(a.g, "calculator",
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
}

func (a *App) defineFlows() {
	// ジョークを生成するFlow（構造化出力付き）
	genkitsdk.DefineFlow(a.g, "tellJoke",
		func(ctx context.Context, input JokeInput) (JokeOutput, error) {
			result, _, err := genkitsdk.GenerateData[JokeOutput](ctx, a.g,
				ai.WithModel(a.model),
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
	genkitsdk.DefineFlow(a.g, "analyzeSentiment",
		func(ctx context.Context, text string) (SentimentOutput, error) {
			result, _, err := genkitsdk.GenerateData[SentimentOutput](ctx, a.g,
				ai.WithModel(a.model),
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
}
