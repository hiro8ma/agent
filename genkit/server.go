package genkit

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"

	"github.com/firebase/genkit/go/ai"
	genkitsdk "github.com/firebase/genkit/go/genkit"
)

// RunServer はREST APIサーバーを起動
func (a *App) RunServer() {
	mux := http.NewServeMux()

	// すべてのFlowをHandlerでエクスポート
	for _, flow := range genkitsdk.ListFlows(a.g) {
		name := flow.Desc().Name
		mux.HandleFunc("POST /api/"+name, genkitsdk.Handler(flow))
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
		response, err := genkitsdk.Generate(ctx, a.g,
			ai.WithModel(a.model),
			ai.WithPrompt(req.Message),
			ai.WithTools(a.calcTool),
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
		if !a.useGemini {
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
		if a.store.Len() == 0 {
			if err := a.indexDocuments(ctx); err != nil {
				http.Error(w, "Failed to index documents: "+err.Error(), http.StatusInternalServerError)
				return
			}
		}

		answer, err := a.ragQuery(ctx, req.Question)
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
