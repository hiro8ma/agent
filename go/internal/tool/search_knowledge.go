package tool

import (
	"context"
	"fmt"

	"github.com/hiro8ma/agent/go/internal/agent/domain/model"
)

// SearchKnowledgeSchema は search_knowledge Tool 宣言。
var SearchKnowledgeSchema = model.ToolSchema{
	Name:        "search_knowledge",
	Description: "AI エンジニアリングに関する知識を検索する。RAG / 評価 / プロンプト / エージェント設計 / ファインチューニングなど",
	Parameters: model.ParameterSchema{
		Type: "object",
		Properties: map[string]model.PropertySchema{
			"query": {Type: "string", Description: "検索クエリ"},
		},
		Required: []string{"query"},
	},
}

// SearchKnowledge は SearchKnowledgeSchema の Handler。
// MVP は stub、将来 mcp/ai_knowledge 経由にする。
func SearchKnowledge(ctx context.Context, args map[string]any) (map[string]any, error) {
	query, _ := args["query"].(string)
	fmt.Printf("  [Tool search_knowledge] query=%q\n", query)
	return map[string]any{
		"result": fmt.Sprintf("AI 知識検索結果 '%s' に関する概要を返す（stub）", query),
	}, nil
}
