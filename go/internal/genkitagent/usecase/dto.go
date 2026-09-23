package usecase

import "github.com/hiro8ma/agent/go/internal/genkitagent/domain/model"

type ListAgentsRequest struct{}

type ListAgentsResponse struct {
	Agents []model.AgentInfo
}

type ChatRequest struct {
	AgentID   string
	SessionID string
	Message   string
}

// ChatResponse は Delta と Output のどちらか一方だけを持つ。Output を持つ応答が列の最後になる。
type ChatResponse struct {
	Delta        *model.ChatChunk
	Output       *model.ChatOutput
	HistorySaved bool
}

type ExecuteConfirmedToolCallRequest struct {
	ToolCallID string
}

type ExecuteConfirmedToolCallResponse struct {
	Result map[string]any
}
