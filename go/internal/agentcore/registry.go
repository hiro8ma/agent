package agentcore

import (
	"context"
	"iter"
)

// AgentInfo はエージェントの公開メタデータ。
type AgentInfo struct {
	ID          string
	Description string
}

// Agent は 1 体のエージェント。genkit / ADK / genai の各実装がこれを満たす。
// Ask はチャンク列と最終出力を 1 本のシーケンスで返し、エラーも AskOutput.ErrorMessage に畳み込む。
type Agent interface {
	Info() AgentInfo
	Ask(ctx context.Context, input *AskInput) iter.Seq2[*AskChunk, *AskOutput]
}

// Registry はエージェントの登録と選択。
type Registry struct {
	agents map[string]Agent
	order  []string
}

func NewRegistry(agents ...Agent) *Registry {
	r := &Registry{agents: map[string]Agent{}}
	for _, a := range agents {
		r.agents[a.Info().ID] = a
		r.order = append(r.order, a.Info().ID)
	}
	return r
}

func (r *Registry) Get(id string) (Agent, bool) {
	a, ok := r.agents[id]
	return a, ok
}

func (r *Registry) List() []AgentInfo {
	infos := make([]AgentInfo, 0, len(r.order))
	for _, id := range r.order {
		infos = append(infos, r.agents[id].Info())
	}
	return infos
}

// SessionStore はセッション単位の履歴ストア。
type SessionStore interface {
	Load(ctx context.Context, sessionID string) ([]Message, error)
	Append(ctx context.Context, sessionID string, messages ...Message) error
}

// ToolExecutor は承認済みツール呼び出しの実行。
// 承認待ちが存在しない（または実行済みの）場合は ErrPendingNotFound を返す。
type ToolExecutor interface {
	Execute(ctx context.Context, toolCallID string) (map[string]any, error)
}
