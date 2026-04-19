// Package externalservice は LLM や外部サービスへの抽象インターフェース。
// 実装は internal/{genai,genkit,adk}/ に置く。
package externalservice

import (
	"context"

	"github.com/hiro8ma/agent/go/internal/agent/domain/model"
)

// LLMResponse は LLM の 1 回分の応答。
type LLMResponse struct {
	Message      model.Message
	TotalTokens  int
	FinishReason string
}

// LLMProvider は会話履歴と Tool 宣言を受け取り、LLM 応答（テキスト or Tool 呼び出し）を返す。
// この境界があることで usecase 層は genai / genkit / adk どの SDK にも依存しない。
type LLMProvider interface {
	Generate(ctx context.Context, systemPrompt string, history []model.Message, tools []model.ToolSchema) (LLMResponse, error)
}
