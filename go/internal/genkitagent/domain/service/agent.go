// Package service は Genkit 版のエージェントに問い合わせるドメインサービスと、その Genkit による実装。
package service

import (
	"context"
	"iter"

	"github.com/hiro8ma/agent/go/internal/genkitagent/domain/model"
)

// Agent はチャンク列と最終出力を 1 本のシーケンスで返し、エラーも ChatOutput.ErrorMessage に畳み込む。
type Agent interface {
	Info() model.AgentInfo
	Chat(ctx context.Context, input *model.ChatInput) iter.Seq2[*model.ChatChunk, *model.ChatOutput]
}

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

func (r *Registry) List() []model.AgentInfo {
	infos := make([]model.AgentInfo, 0, len(r.order))
	for _, id := range r.order {
		infos = append(infos, r.agents[id].Info())
	}
	return infos
}
