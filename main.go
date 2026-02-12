package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"sort"
	"sync"

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

// ========================================
// インメモリベクトルストア（RAG用）
// ========================================

// VectorDocument はベクトル化されたドキュメント
type VectorDocument struct {
	ID        string
	Content   string
	Embedding []float32
	Metadata  map[string]string
}

// VectorStore はインメモリのベクトルストア
type VectorStore struct {
	mu        sync.RWMutex
	documents []VectorDocument
}

// NewVectorStore は新しいベクトルストアを作成
func NewVectorStore() *VectorStore {
	return &VectorStore{
		documents: make([]VectorDocument, 0),
	}
}

// Add はドキュメントをストアに追加
func (vs *VectorStore) Add(doc VectorDocument) {
	vs.mu.Lock()
	defer vs.mu.Unlock()
	vs.documents = append(vs.documents, doc)
}

// Search はコサイン類似度で類似ドキュメントを検索
func (vs *VectorStore) Search(queryEmbedding []float32, topK int) []VectorDocument {
	vs.mu.RLock()
	defer vs.mu.RUnlock()

	type scored struct {
		doc   VectorDocument
		score float64
	}

	var results []scored
	for _, doc := range vs.documents {
		score := cosineSimilarity(queryEmbedding, doc.Embedding)
		results = append(results, scored{doc: doc, score: score})
	}

	// スコア降順でソート
	sort.Slice(results, func(i, j int) bool {
		return results[i].score > results[j].score
	})

	// 上位K件を返す
	var topDocs []VectorDocument
	for i := 0; i < topK && i < len(results); i++ {
		topDocs = append(topDocs, results[i].doc)
	}
	return topDocs
}

// cosineSimilarity はコサイン類似度を計算
func cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) {
		return 0
	}
	var dotProduct, normA, normB float64
	for i := range a {
		dotProduct += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}

// グローバル変数（サーバーモードで使用）
var (
	genkitInstance *genkit.Genkit
	defaultModel   ai.Model
	calcTool       ai.Tool
	vectorStore    *VectorStore
	embedder       ai.Embedder
	useGemini      bool
)

func main() {
	ctx := context.Background()

	// 動作モードの確認
	mode := os.Getenv("MODE")
	if mode == "" {
		mode = "demo" // デフォルトはデモモード
	}

	// APIキーの確認（Gemini使用時）
	useGemini = os.Getenv("GEMINI_API_KEY") != ""
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
		embedder = googlegenai.GoogleAIEmbedder(genkitInstance, "text-embedding-004")
		fmt.Println("Using: Google AI (Gemini) with Embedder")
	} else {
		defaultModel = ollama.Model(genkitInstance, "llama3.2")
		fmt.Println("Using: Ollama (llama3.2) - RAG disabled (no embedder)")
	}

	// ベクトルストアの初期化
	vectorStore = NewVectorStore()

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

// ========================================
// RAG関連関数
// ========================================

// indexDocuments はサンプルドキュメントをベクトル化してインデックス
func indexDocuments(ctx context.Context) error {
	if embedder == nil {
		return fmt.Errorf("embedder is not available")
	}

	// サンプルナレッジベース（会社のFAQ風）
	documents := []struct {
		id      string
		content string
	}{
		{"doc1", "Genkit Goは、Googleが開発したAIアプリケーション開発フレームワークです。Go言語で型安全にAIアプリを構築できます。"},
		{"doc2", "Genkitの主な機能には、テキスト生成、構造化出力、Tool Calling、Flow、RAGがあります。"},
		{"doc3", "Tool Callingを使うと、LLMに外部関数を実行させることができます。計算、API呼び出し、データベースアクセスなどが可能です。"},
		{"doc4", "Flowは再利用可能なワークフローを定義する機能です。テスト、監視、デプロイが容易になります。"},
		{"doc5", "RAG（Retrieval-Augmented Generation）は、外部知識を検索してLLMの回答を強化する技術です。"},
		{"doc6", "Genkitは複数のモデルプロバイダをサポートしています。Google AI、Vertex AI、OpenAI、Anthropic、Ollamaなどが使えます。"},
		{"doc7", "構造化出力を使うと、LLMの出力をGoの構造体に直接マッピングできます。JSONスキーマで型安全性を確保します。"},
		{"doc8", "Genkitのサーバーモードでは、FlowをREST APIエンドポイントとして公開できます。genkit.Handler()を使います。"},
	}

	fmt.Println("  ドキュメントをインデックス中...")
	for _, doc := range documents {
		// Embeddingを生成
		res, err := genkit.Embed(ctx, genkitInstance,
			ai.WithEmbedder(embedder),
			ai.WithTextDocs(doc.content),
		)
		if err != nil {
			return fmt.Errorf("embedding failed for %s: %w", doc.id, err)
		}

		if len(res.Embeddings) == 0 {
			return fmt.Errorf("no embedding returned for %s", doc.id)
		}

		// ベクトルストアに追加
		vectorStore.Add(VectorDocument{
			ID:        doc.id,
			Content:   doc.content,
			Embedding: res.Embeddings[0].Embedding,
		})
		fmt.Printf("    Indexed: %s\n", doc.id)
	}

	return nil
}

// ragQuery はRAGを使って質問に回答
func ragQuery(ctx context.Context, question string) (string, error) {
	if embedder == nil {
		return "", fmt.Errorf("RAG is not available (embedder not configured)")
	}

	// 質問をベクトル化
	res, err := genkit.Embed(ctx, genkitInstance,
		ai.WithEmbedder(embedder),
		ai.WithTextDocs(question),
	)
	if err != nil {
		return "", fmt.Errorf("question embedding failed: %w", err)
	}

	if len(res.Embeddings) == 0 {
		return "", fmt.Errorf("no embedding returned for question")
	}

	// 類似ドキュメントを検索（上位3件）
	relevantDocs := vectorStore.Search(res.Embeddings[0].Embedding, 3)

	// コンテキストを構築
	var context string
	for i, doc := range relevantDocs {
		context += fmt.Sprintf("[%d] %s\n", i+1, doc.Content)
	}

	// RAGプロンプトで回答生成
	prompt := fmt.Sprintf(`以下の参考情報を元に、質問に答えてください。

参考情報:
%s

質問: %s

参考情報に基づいて、簡潔に回答してください。参考情報にない内容は「情報がありません」と答えてください。`, context, question)

	answer, err := genkit.GenerateText(ctx, genkitInstance,
		ai.WithModel(defaultModel),
		ai.WithPrompt(prompt),
	)
	if err != nil {
		return "", fmt.Errorf("generation failed: %w", err)
	}

	return answer, nil
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

	// === デモ5: RAG（知識検索型生成）===
	if useGemini {
		fmt.Println("\n=== デモ5: RAG（知識検索型生成）===")

		// ドキュメントをインデックス
		if err := indexDocuments(ctx); err != nil {
			log.Printf("Failed to index documents: %v", err)
		} else {
			// RAGで質問に回答
			questions := []string{
				"Genkitとは何ですか？",
				"Tool Callingで何ができますか？",
				"構造化出力の利点は？",
			}

			for _, q := range questions {
				fmt.Printf("\n質問: %s\n", q)
				answer, err := ragQuery(ctx, q)
				if err != nil {
					log.Printf("RAG query failed: %v", err)
				} else {
					fmt.Printf("回答: %s\n", answer)
				}
			}
		}
	} else {
		fmt.Println("\n=== デモ5: RAG（スキップ - Gemini APIが必要）===")
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

	// RAGエンドポイント（Gemini使用時のみ）
	mux.HandleFunc("POST /api/rag", func(w http.ResponseWriter, r *http.Request) {
		if !useGemini {
			http.Error(w, "RAG requires Gemini API", http.StatusServiceUnavailable)
			return
		}

		var req struct {
			Question string `json:"question"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid JSON", http.StatusBadRequest)
			return
		}

		ctx := r.Context()

		// 初回はドキュメントをインデックス
		if len(vectorStore.documents) == 0 {
			if err := indexDocuments(ctx); err != nil {
				http.Error(w, "Failed to index documents: "+err.Error(), http.StatusInternalServerError)
				return
			}
		}

		answer, err := ragQuery(ctx, req.Question)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{
			"answer": answer,
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
	fmt.Println("  POST /api/tellJoke         - ジョーク生成 {\"data\": {\"topic\": \"テーマ\"}}")
	fmt.Println("  POST /api/analyzeSentiment - 感情分析 {\"data\": \"テキスト\"}")
	fmt.Println("  POST /api/chat             - チャット {\"message\": \"メッセージ\"}")
	fmt.Println("  POST /api/rag              - RAG検索 {\"question\": \"質問\"}")
	fmt.Println("  GET  /health               - ヘルスチェック")
	fmt.Println()

	if err := http.ListenAndServe(":"+port, mux); err != nil {
		log.Fatal(err)
	}
}
