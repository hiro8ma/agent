package usecase

import (
	"context"
	"fmt"

	"github.com/hiro8ma/agent/go/internal/agent/domain/externalservice"
	"github.com/hiro8ma/agent/go/internal/agent/domain/model"
)

// AskConfig はユースケース実装のパラメータ。
type AskConfig struct {
	SystemPrompt     string
	MaxIterations    int
	MaxTokensPerTurn int
}

// DefaultAskConfig は推奨値。
func DefaultAskConfig(systemPrompt string) AskConfig {
	return AskConfig{
		SystemPrompt:     systemPrompt,
		MaxIterations:    10,
		MaxTokensPerTurn: 128_000,
	}
}

type askImpl struct {
	cfg  AskConfig
	llm  externalservice.LLMProvider
	tool externalservice.ToolExecutor
}

// NewAsk は Ask ユースケース実装を生成する。
func NewAsk(cfg AskConfig, llm externalservice.LLMProvider, tool externalservice.ToolExecutor) Ask {
	return &askImpl{cfg: cfg, llm: llm, tool: tool}
}

// Handle は Tool Dispatch ループを回す。LLM の応答に ToolCall がある限り実行結果を履歴に積んで再度呼ぶ。
// LLM が最終テキストを返した時点で終了する。
// この実装は genai / genkit / adk のどれでも LLMProvider が実装されていれば動く。
func (a *askImpl) Handle(ctx context.Context, in AskInput) (AskOutput, error) {
	conv := in.Conversation
	if conv == nil {
		conv = &model.Conversation{}
	}
	conv.Append(model.Message{Role: model.RoleUser, Text: in.UserMessage})

	tools := a.tool.Schemas()
	totalTokens := 0

	for iter := 0; iter < a.cfg.MaxIterations; iter++ {
		resp, err := a.llm.Generate(ctx, a.cfg.SystemPrompt, conv.Messages, tools)
		if err != nil {
			return AskOutput{}, fmt.Errorf("llm generate (iter=%d) %w", iter, err)
		}

		totalTokens += resp.TotalTokens
		if totalTokens > a.cfg.MaxTokensPerTurn {
			return AskOutput{}, fmt.Errorf("token 累積上限に到達 %d > %d", totalTokens, a.cfg.MaxTokensPerTurn)
		}

		conv.Append(resp.Message)

		if len(resp.Message.ToolCalls) == 0 {
			return AskOutput{
				Answer:       resp.Message.Text,
				Conversation: conv,
				Iterations:   iter + 1,
				TotalTokens:  totalTokens,
			}, nil
		}

		for _, call := range resp.Message.ToolCalls {
			result := a.tool.Execute(ctx, call)
			conv.Append(model.Message{
				Role:       model.RoleTool,
				ToolResult: &result,
			})
		}
	}

	return AskOutput{Conversation: conv, Iterations: a.cfg.MaxIterations, TotalTokens: totalTokens},
		fmt.Errorf("iteration 上限 %d に到達", a.cfg.MaxIterations)
}
