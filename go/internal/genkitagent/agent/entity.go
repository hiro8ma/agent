package agent

import "github.com/hiro8ma/agent/go/internal/agentcore"

// 入出力型とドメイン型は agentcore（フレームワーク非依存の核）のエイリアス。
// genkit / ADK / genai の 3 実装で同じ型を使い、transport（Connect ハンドラ）を共有する。
type (
	AskInput          = agentcore.AskInput
	Message           = agentcore.Message
	AskChunk          = agentcore.AskChunk
	AskOutput         = agentcore.AskOutput
	PendingToolCall   = agentcore.PendingToolCall
	ToolCall          = agentcore.ToolCall
	TokenUsage        = agentcore.TokenUsage
	Order             = agentcore.Order
	KnowledgeDoc      = agentcore.KnowledgeDoc
	OrderService      = agentcore.OrderService
	GeoService        = agentcore.GeoService
	KnowledgeSearcher = agentcore.KnowledgeSearcher
)
