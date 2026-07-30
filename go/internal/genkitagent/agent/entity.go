package agent

// AskInput は 1 回の問い合わせ。History はセッションストアから復元した過去メッセージ。
type AskInput struct {
	SessionID   string
	UserMessage string
	History     []Message
}

// Message はセッション履歴の 1 メッセージ。
type Message struct {
	Role string `firestore:"role" json:"role"` // "user" | "model"
	Text string `firestore:"text" json:"text"`
}

// AskChunk はストリーミング中の増分。
type AskChunk struct {
	AnswerDelta string `json:"answerDelta"`
}

// AskOutput は最終応答。
type AskOutput struct {
	SessionID        string            `json:"sessionId"`
	Answer           string            `json:"answer"`
	FinishReason     string            `json:"finishReason"`
	ToolCalls        []ToolCall        `json:"toolCalls,omitempty"`
	PendingToolCalls []PendingToolCall `json:"pendingToolCalls,omitempty"`
	Usage            TokenUsage        `json:"usage"`
	ErrorMessage     string            `json:"errorMessage,omitempty"`
}

// PendingToolCall は承認待ちのツール呼び出し。ExecuteConfirmedToolCall で実行される。
type PendingToolCall struct {
	ID    string         `firestore:"-"     json:"toolCallId"`
	Name  string         `firestore:"name"  json:"name"`
	Input map[string]any `firestore:"input" json:"input,omitempty"`
}

// ToolCall はモデルが実行したツール呼び出しの記録。
type ToolCall struct {
	Name  string `json:"name"`
	Input any    `json:"input,omitempty"`
}

// TokenUsage はトークン消費量。
type TokenUsage struct {
	InputTokens  int `json:"inputTokens"`
	OutputTokens int `json:"outputTokens"`
	TotalTokens  int `json:"totalTokens"`
}
