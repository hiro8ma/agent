// Package tool は 3 実装（genai / genkit / adk）で共有する Tool 定義・Handler。
package tool

import (
	"context"
	"fmt"

	"github.com/hiro8ma/agent/go/internal/agent/domain/model"
)

// Handler は 1 Tool の実行関数。
type Handler func(ctx context.Context, args map[string]any) (map[string]any, error)

// Registry は複数 Tool を束ねる ToolExecutor 実装。
type Registry struct {
	schemas  []model.ToolSchema
	handlers map[string]Handler
}

// NewRegistry は空のレジストリを生成する。
func NewRegistry() *Registry {
	return &Registry{handlers: map[string]Handler{}}
}

// Register は Tool を追加する。
func (r *Registry) Register(schema model.ToolSchema, handler Handler) {
	r.schemas = append(r.schemas, schema)
	r.handlers[schema.Name] = handler
}

// Schemas は登録済み Tool 宣言を返す。
func (r *Registry) Schemas() []model.ToolSchema {
	return r.schemas
}

// Execute は ToolCall を対応 Handler にディスパッチする。
func (r *Registry) Execute(ctx context.Context, call model.ToolCall) model.ToolResult {
	h, ok := r.handlers[call.Name]
	if !ok {
		return model.ToolResult{CallID: call.ID, Name: call.Name, Err: fmt.Sprintf("未定義の Tool %q", call.Name)}
	}
	payload, err := h(ctx, call.Args)
	if err != nil {
		return model.ToolResult{CallID: call.ID, Name: call.Name, Err: err.Error()}
	}
	return model.ToolResult{CallID: call.ID, Name: call.Name, Payload: payload}
}

// Default はデモ用の Tool セット（calculator + search_knowledge）を事前登録したレジストリを返す。
func Default() *Registry {
	r := NewRegistry()
	r.Register(CalculatorSchema, Calculator)
	r.Register(SearchKnowledgeSchema, SearchKnowledge)
	return r
}
