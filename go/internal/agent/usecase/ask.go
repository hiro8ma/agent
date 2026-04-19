// Package usecase はアプリケーションサービス層。
// LLM / Tool / 履歴管理の組み合わせを実現する。
package usecase

import (
	"context"

	"github.com/hiro8ma/agent/go/internal/agent/domain/model"
)

// AskInput はユースケース入力 DTO。
type AskInput struct {
	UserMessage  string
	Conversation *model.Conversation
}

// AskOutput はユースケース出力 DTO。
type AskOutput struct {
	Answer       string
	Conversation *model.Conversation
	Iterations   int
	TotalTokens  int
}

// Ask は 1 ターンの問い合わせを受け付け、Tool Dispatch ループを回して応答を返す。
type Ask interface {
	Handle(ctx context.Context, in AskInput) (AskOutput, error)
}
