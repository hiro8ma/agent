package workflow

import (
	"context"

	"github.com/hiro8ma/agent/go/internal/agent/domain/model"
	"github.com/hiro8ma/agent/go/internal/agent/usecase"
	"github.com/hiro8ma/agent/go/internal/lib/liberrors"
)

// LLMActivity は usecase.Ask（genkit / genai / adk 実装）を Activity として包む。
// WHY: フェーズ内の LLM 往復を Activity の単位に閉じ込めることで、フェーズ遷移と耐久性は
// エンジンが、1 フェーズの推論は既存実装が担う二層構造になる。
// ActivityInput.Messages の先頭をユーザー発話として扱う。
type LLMActivity struct {
	ActivityName string
	Ask          usecase.Ask
}

// Name は Activity を満たす。
func (a LLMActivity) Name() string { return a.ActivityName }

// Execute は Activity を満たす。失敗時はエンジンが RetryPolicy に従って再試行する。
func (a LLMActivity) Execute(ctx context.Context, in ActivityInput) (ActivityResult, error) {
	prompt := in.Operation
	if len(in.Messages) > 0 {
		prompt = in.Messages[0]
	}
	out, err := a.Ask.Handle(ctx, usecase.AskInput{
		UserMessage:  prompt,
		Conversation: &model.Conversation{},
	})
	if err != nil {
		return ActivityResult{}, liberrors.Wrap(liberrors.CodeUnavailable, err, "llm activity %q", a.ActivityName)
	}
	return ActivityResult{Output: map[string]any{
		"answer":       out.Answer,
		"iterations":   out.Iterations,
		"total_tokens": out.TotalTokens,
	}}, nil
}
