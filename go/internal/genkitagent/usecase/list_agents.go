package usecase

import "context"

func (s *AgentService) ListAgents(_ context.Context, _ *ListAgentsRequest) (*ListAgentsResponse, error) {
	return &ListAgentsResponse{Agents: s.agents.List()}, nil
}
