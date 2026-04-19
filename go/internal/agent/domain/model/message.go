// Package model はエージェントのドメインモデル。
// 特定の LLM SDK（genai / genkit / adk）に依存しない表現にする。
package model

// Role はメッセージの発信主体。
type Role string

const (
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
	RoleSystem    Role = "system"
)

// Message は会話の 1 ターン分のやり取り。
type Message struct {
	Role       Role
	Text       string
	ToolCalls  []ToolCall
	ToolResult *ToolResult
}

// ToolCall は LLM が要求する Tool 呼び出し。
type ToolCall struct {
	ID   string
	Name string
	Args map[string]any
}

// ToolResult は Tool 実行結果。
type ToolResult struct {
	CallID  string
	Name    string
	Payload map[string]any
	Err     string
}

// Conversation は会話履歴を保持する。
type Conversation struct {
	Messages []Message
}

// Append は履歴に 1 メッセージ追加する。
func (c *Conversation) Append(m Message) {
	c.Messages = append(c.Messages, m)
}
