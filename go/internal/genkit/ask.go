// Package genkit は usecase.Ask を Firebase Genkit で実装する。
// Genkit は内部で Tool Dispatch ループを持つため、LLMProvider ではなく
// Ask IF を直接実装する形にする（genai の step-loop と対比できる）。
package genkit

import (
	"context"
	"fmt"

	"github.com/firebase/genkit/go/ai"
	genkitsdk "github.com/firebase/genkit/go/genkit"

	"github.com/hiro8ma/agent/go/internal/agent/domain/model"
	"github.com/hiro8ma/agent/go/internal/agent/usecase"
	"github.com/hiro8ma/agent/go/internal/lib/liberrors"
)

// ToolHandler は 1 Tool の実行関数。
type ToolHandler func(ctx context.Context, args map[string]any) (map[string]any, error)

// ToolSpec は Tool のスキーマとハンドラをペアにしたもの。
type ToolSpec struct {
	Schema  model.ToolSchema
	Handler ToolHandler
}

// Ask は Genkit Generate を 1 回呼び、内部の Tool ループに任せる実装。
type Ask struct {
	g            *genkitsdk.Genkit
	systemPrompt string
	model        ai.Model
	tools        []ai.ToolRef
}

// New は Genkit ベースの Ask 実装を生成する。
// tools は model.ToolSchema と Handler のペア。Genkit の DefineTool に変換する。
func New(g *genkitsdk.Genkit, model ai.Model, systemPrompt string, tools []ToolSpec) *Ask {
	refs := make([]ai.ToolRef, 0, len(tools))
	for _, t := range tools {
		refs = append(refs, defineTool(g, t))
	}
	return &Ask{g: g, systemPrompt: systemPrompt, model: model, tools: refs}
}

// Handle は会話履歴 + Tool 宣言を Genkit に渡し、最終応答を返す。
// Tool Dispatch ループは Genkit が内部で回すため、ここにループは書かない。
func (a *Ask) Handle(ctx context.Context, in usecase.AskInput) (usecase.AskOutput, error) {
	conv := in.Conversation
	if conv == nil {
		conv = &model.Conversation{}
	}
	conv.Append(model.Message{Role: model.RoleUser, Text: in.UserMessage})

	messages, err := toGenkitMessages(conv.Messages)
	if err != nil {
		return usecase.AskOutput{}, liberrors.Wrap(liberrors.CodeInvalidArgument, err, "messages convert")
	}

	opts := []ai.GenerateOption{
		ai.WithModel(a.model),
		ai.WithSystem(a.systemPrompt),
		ai.WithMessages(messages...),
	}
	for _, t := range a.tools {
		opts = append(opts, ai.WithTools(t))
	}

	resp, err := genkitsdk.Generate(ctx, a.g, opts...)
	if err != nil {
		return usecase.AskOutput{}, liberrors.Wrap(liberrors.CodeUnavailable, err, "genkit generate")
	}

	answer := resp.Text()
	conv.Append(model.Message{Role: model.RoleAssistant, Text: answer})

	totalTokens := 0
	if resp.Usage != nil {
		totalTokens = resp.Usage.TotalTokens
	}

	return usecase.AskOutput{
		Answer:       answer,
		Conversation: conv,
		Iterations:   1, // Genkit は内部でループする。外からは 1 回に見える
		TotalTokens:  totalTokens,
	}, nil
}

// ---------- 変換 ----------

func defineTool(g *genkitsdk.Genkit, spec ToolSpec) ai.Tool {
	return genkitsdk.DefineTool(g, spec.Schema.Name, spec.Schema.Description,
		func(tc *ai.ToolContext, args map[string]any) (map[string]any, error) {
			return spec.Handler(tc, args)
		},
	)
}

func toGenkitMessages(history []model.Message) ([]*ai.Message, error) {
	var out []*ai.Message
	for _, m := range history {
		switch m.Role {
		case model.RoleUser:
			out = append(out, ai.NewUserTextMessage(m.Text))
		case model.RoleAssistant:
			if m.Text != "" {
				out = append(out, ai.NewModelTextMessage(m.Text))
			}
		case model.RoleTool:
			// Genkit は Tool ループを内部で完結させる設計のため、外から tool result を注入する表現は使わない。
			// ユースケースの再開（途中から会話を復元する）が必要になった段階で別途実装する。
			return nil, fmt.Errorf("genkit adapter は外部注入の tool result 復元を未対応（未来ターンのみ扱う想定）")
		}
	}
	return out, nil
}
