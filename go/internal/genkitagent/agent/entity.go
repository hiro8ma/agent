package agent

import "github.com/hiro8ma/agent/go/internal/agentcore"

// 入出力型とドメイン型は agentcore（フレームワーク非依存の核）のエイリアス。
// genkit / ADK / genai の 3 実装で同じ型を使い、transport（Connect ハンドラ）を共有する。
type (
	ChatInput         = agentcore.ChatInput
	Message           = agentcore.Message
	ChatChunk         = agentcore.ChatChunk
	ChatOutput        = agentcore.ChatOutput
	PendingToolCall   = agentcore.PendingToolCall
	ToolCall          = agentcore.ToolCall
	TokenUsage        = agentcore.TokenUsage
	Order             = agentcore.Order
	KnowledgeDoc      = agentcore.KnowledgeDoc
	OrderService      = agentcore.OrderService
	GeoService        = agentcore.GeoService
	KnowledgeSearcher = agentcore.KnowledgeSearcher
)
