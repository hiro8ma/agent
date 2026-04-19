package externalservice

import (
	"context"

	"github.com/hiro8ma/agent/go/internal/agent/domain/model"
)

// ToolExecutor は Tool の実行を担う。
// LLM から受け取った ToolCall を処理し、ToolResult を返す。
type ToolExecutor interface {
	Schemas() []model.ToolSchema
	Execute(ctx context.Context, call model.ToolCall) model.ToolResult
}
