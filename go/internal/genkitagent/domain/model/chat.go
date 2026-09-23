package model

import "errors"

type ChatInput struct {
	SessionID   string
	UserMessage string
	History     []Message
}

type Message struct {
	Role string `json:"role"`
	Text string `json:"text"`
}

const (
	RoleUser  = "user"
	RoleModel = "model"
)

type ChatChunk struct {
	AnswerDelta string `json:"answerDelta"`
}

type ChatOutput struct {
	SessionID        string            `json:"sessionId"`
	Answer           string            `json:"answer"`
	FinishReason     string            `json:"finishReason"`
	ToolCalls        []ToolCall        `json:"toolCalls,omitempty"`
	PendingToolCalls []PendingToolCall `json:"pendingToolCalls,omitempty"`
	Usage            TokenUsage        `json:"usage"`
	ErrorMessage     string            `json:"errorMessage,omitempty"`
}

func (o *ChatOutput) Failed() bool {
	return o.ErrorMessage != ""
}

type PendingToolCall struct {
	ID    string         `json:"toolCallId"`
	Name  string         `json:"name"`
	Input map[string]any `json:"input,omitempty"`
}

type ToolCall struct {
	Name  string `json:"name"`
	Input any    `json:"input,omitempty"`
}

type TokenUsage struct {
	InputTokens  int `json:"inputTokens"`
	OutputTokens int `json:"outputTokens"`
	TotalTokens  int `json:"totalTokens"`
}

// Total は TotalTokens を返さないモデルでも、入出力の和で消費量を数えられるようにする。
func (u TokenUsage) Total() int {
	if u.TotalTokens > 0 {
		return u.TotalTokens
	}
	return u.InputTokens + u.OutputTokens
}

var ErrPendingNotFound = errors.New("pending tool call not found")
