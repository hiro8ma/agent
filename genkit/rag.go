package genkit

import (
	"context"
	"fmt"

	"github.com/firebase/genkit/go/ai"
	genkitsdk "github.com/firebase/genkit/go/genkit"
)

// indexDocuments はサンプルドキュメントをベクトル化してインデックス
func (a *App) indexDocuments(ctx context.Context) error {
	if a.embedder == nil {
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
		res, err := genkitsdk.Embed(ctx, a.g,
			ai.WithEmbedder(a.embedder),
			ai.WithTextDocs(doc.content),
		)
		if err != nil {
			return fmt.Errorf("embedding failed for %s: %w", doc.id, err)
		}

		if len(res.Embeddings) == 0 {
			return fmt.Errorf("no embedding returned for %s", doc.id)
		}

		// ベクトルストアに追加
		a.store.Add(VectorDocument{
			ID:        doc.id,
			Content:   doc.content,
			Embedding: res.Embeddings[0].Embedding,
		})
		fmt.Printf("    Indexed: %s\n", doc.id)
	}

	return nil
}

// ragQuery はRAGを使って質問に回答
func (a *App) ragQuery(ctx context.Context, question string) (string, error) {
	if a.embedder == nil {
		return "", fmt.Errorf("RAG is not available (embedder not configured)")
	}

	// 質問をベクトル化
	res, err := genkitsdk.Embed(ctx, a.g,
		ai.WithEmbedder(a.embedder),
		ai.WithTextDocs(question),
	)
	if err != nil {
		return "", fmt.Errorf("question embedding failed: %w", err)
	}

	if len(res.Embeddings) == 0 {
		return "", fmt.Errorf("no embedding returned for question")
	}

	// 類似ドキュメントを検索（上位3件）
	relevantDocs := a.store.Search(res.Embeddings[0].Embedding, 3)

	// コンテキストを構築
	var ragContext string
	for i, doc := range relevantDocs {
		ragContext += fmt.Sprintf("[%d] %s\n", i+1, doc.Content)
	}

	// RAGプロンプトで回答生成
	prompt := fmt.Sprintf(`以下の参考情報を元に、質問に答えてください。

参考情報:
%s
質問: %s

参考情報に基づいて、簡潔に回答してください。参考情報にない内容は「情報がありません」と答えてください。`, ragContext, question)

	answer, err := genkitsdk.GenerateText(ctx, a.g,
		ai.WithModel(a.model),
		ai.WithPrompt(prompt),
	)
	if err != nil {
		return "", fmt.Errorf("generation failed: %w", err)
	}

	return answer, nil
}
