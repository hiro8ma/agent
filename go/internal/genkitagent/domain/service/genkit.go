package service

import (
	"context"
	"iter"
	"strings"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/middleware"

	"github.com/hiro8ma/agent/go/internal/genkitagent/domain/model"
)

const (
	chatMaxTurns       = 5
	generateMaxRetries = 2
	baseBackoff        = 2 * time.Second
	maxBackoff         = 30 * time.Second
)

// Definition はエージェント 1 体の定義。部署・役割ごとに system prompt とツールを変えて量産する。
type Definition struct {
	ID           string
	Description  string
	SystemPrompt string
	Tools        []ai.ToolRef
	// SkillPaths は Agent Skills（SKILL.md を持つディレクトリの親）を探すパス。
	// 空のときは skills ミドルウェアを差し込まない。
	SkillPaths []string
	// Use は Generate に差し込むミドルウェア。ツールの権限の検査など。
	Use []ai.Middleware
}

// GenkitAgent は 1 つの Definition を Genkit の streaming flow として公開する。
type GenkitAgent struct {
	def  Definition
	flow *core.Flow[*model.ChatInput, *model.ChatOutput, *model.ChatChunk]
}

var _ Agent = (*GenkitAgent)(nil)

func NewGenkitAgent(g *genkit.Genkit, def Definition) *GenkitAgent {
	return &GenkitAgent{
		def: def,
		flow: genkit.DefineStreamingFlow(g, "chat-"+def.ID,
			func(ctx context.Context, input *model.ChatInput, cb core.StreamCallback[*model.ChatChunk]) (*model.ChatOutput, error) {
				return runChat(ctx, g, def, input, cb)
			},
		),
	}
}

func (a *GenkitAgent) Definition() Definition { return a.def }

func (a *GenkitAgent) Info() model.AgentInfo {
	return model.AgentInfo{ID: a.def.ID, Description: a.def.Description}
}

func (a *GenkitAgent) Chat(ctx context.Context, input *model.ChatInput) iter.Seq2[*model.ChatChunk, *model.ChatOutput] {
	// Flow は入力を JSON スキーマで検査し、nil のスライスを null として配列の型違いで拒否する。
	if input.History == nil {
		in := *input
		in.History = []model.Message{}
		input = &in
	}
	return func(yield func(*model.ChatChunk, *model.ChatOutput) bool) {
		for v, err := range a.flow.Stream(ctx, input) {
			if err != nil {
				yield(nil, &model.ChatOutput{
					SessionID:    input.SessionID,
					FinishReason: "error",
					ErrorMessage: err.Error(),
				})
				return
			}
			if v.Done {
				yield(nil, v.Output)
				return
			}
			if !yield(v.Stream, nil) {
				return
			}
		}
	}
}

func runChat(ctx context.Context, g *genkit.Genkit, def Definition, input *model.ChatInput, cb core.StreamCallback[*model.ChatChunk]) (*model.ChatOutput, error) {
	messages := assembleMessages(def.SystemPrompt, input)

	streamedAny := false
	stream := func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
		text := chunk.Text()
		if text == "" {
			return nil
		}
		if err := cb(ctx, &model.ChatChunk{AnswerDelta: text}); err != nil {
			return err
		}
		streamedAny = true
		return nil
	}

	opts := []ai.GenerateOption{
		ai.WithMessages(messages...),
		ai.WithTools(def.Tools...),
		ai.WithMaxTurns(chatMaxTurns),
		ai.WithStreaming(stream),
	}
	if len(def.SkillPaths) > 0 {
		// メタデータだけ system prompt に注入し、本文は use_skill 呼び出し時にロードされる。
		opts = append(opts, ai.WithUse(&middleware.Skills{SkillPaths: def.SkillPaths}))
	}
	if len(def.Use) > 0 {
		opts = append(opts, ai.WithUse(def.Use...))
	}

	var resp *ai.ModelResponse
	var err error
	for attempt := 0; ; attempt++ {
		resp, err = genkit.Generate(ctx, g, opts...)
		if err == nil {
			break
		}
		// チャンク送出後にリトライすると先頭から再送して重複するため、未送出時のみリトライする。
		if attempt >= generateMaxRetries || streamedAny || !isResourceExhausted(err) {
			return nil, err
		}
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(backoff(attempt)):
		}
	}

	out := &model.ChatOutput{
		SessionID:        input.SessionID,
		Answer:           resp.Text(),
		FinishReason:     string(resp.FinishReason),
		ToolCalls:        extractToolCalls(resp),
		PendingToolCalls: extractPendingToolCalls(resp),
	}
	if resp.Usage != nil {
		out.Usage = model.TokenUsage{
			InputTokens:  resp.Usage.InputTokens,
			OutputTokens: resp.Usage.OutputTokens,
			TotalTokens:  resp.Usage.TotalTokens,
		}
	}
	return out, nil
}

func assembleMessages(systemPrompt string, input *model.ChatInput) []*ai.Message {
	messages := []*ai.Message{ai.NewSystemTextMessage(systemPrompt)}
	for _, m := range input.History {
		if m.Role == model.RoleModel {
			messages = append(messages, ai.NewModelTextMessage(m.Text))
			continue
		}
		messages = append(messages, ai.NewUserTextMessage(m.Text))
	}
	return append(messages, ai.NewUserTextMessage(input.UserMessage))
}

func extractToolCalls(resp *ai.ModelResponse) []model.ToolCall {
	if resp == nil || resp.Request == nil {
		return nil
	}
	var calls []model.ToolCall
	for _, m := range resp.Request.Messages {
		for _, p := range m.Content {
			if p.IsToolRequest() {
				calls = append(calls, model.ToolCall{Name: p.ToolRequest.Name, Input: p.ToolRequest.Input})
			}
		}
	}
	return calls
}

// extractPendingToolCalls は履歴中のツール応答から承認待ちを拾い上げる。
func extractPendingToolCalls(resp *ai.ModelResponse) []model.PendingToolCall {
	if resp == nil || resp.Request == nil {
		return nil
	}
	var pendings []model.PendingToolCall
	for _, m := range resp.Request.Messages {
		for _, p := range m.Content {
			if !p.IsToolResponse() {
				continue
			}
			out, ok := p.ToolResponse.Output.(map[string]any)
			if !ok {
				continue
			}
			if pending, ok := pendingFromResult(p.ToolResponse.Name, out); ok {
				pendings = append(pendings, pending)
			}
		}
	}
	return pendings
}

// gemini は Dynamic Shared Quota のため一時的に 429 を返すことがある。
func isResourceExhausted(err error) bool {
	if err == nil {
		return false
	}
	msg := err.Error()
	return strings.Contains(msg, "RESOURCE_EXHAUSTED") || strings.Contains(msg, "Error 429")
}

func backoff(attempt int) time.Duration {
	d := baseBackoff << attempt
	if d > maxBackoff {
		return maxBackoff
	}
	return d
}
