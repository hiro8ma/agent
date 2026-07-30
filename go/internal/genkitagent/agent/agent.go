package agent

import (
	"context"
	"iter"
	"strings"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/genkit"
)

const (
	askMaxTurns        = 5
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
}

// Agent は 1 つの Definition を Genkit の streaming flow として公開する。
type Agent struct {
	def  Definition
	flow *core.Flow[*AskInput, *AskOutput, *AskChunk]
}

func New(g *genkit.Genkit, def Definition) *Agent {
	return &Agent{
		def: def,
		flow: genkit.DefineStreamingFlow(g, "ask-"+def.ID,
			func(ctx context.Context, input *AskInput, cb core.StreamCallback[*AskChunk]) (*AskOutput, error) {
				return runAsk(ctx, g, def, input, cb)
			},
		),
	}
}

func (a *Agent) Definition() Definition { return a.def }

// Registry はエージェントの登録と選択。
type Registry struct {
	agents map[string]*Agent
	order  []string
}

func NewRegistry(agents ...*Agent) *Registry {
	r := &Registry{agents: map[string]*Agent{}}
	for _, a := range agents {
		r.agents[a.def.ID] = a
		r.order = append(r.order, a.def.ID)
	}
	return r
}

func (r *Registry) Get(id string) (*Agent, bool) {
	a, ok := r.agents[id]
	return a, ok
}

func (r *Registry) List() []Definition {
	defs := make([]Definition, 0, len(r.order))
	for _, id := range r.order {
		defs = append(defs, r.agents[id].def)
	}
	return defs
}

// Ask はチャンク列と最終出力を 1 本のシーケンスで返す。
// エラーも AskOutput.ErrorMessage に畳み込み、呼び出し側の分岐を 1 箇所にする。
func (a *Agent) Ask(ctx context.Context, input *AskInput) iter.Seq2[*AskChunk, *AskOutput] {
	return func(yield func(*AskChunk, *AskOutput) bool) {
		for v, err := range a.flow.Stream(ctx, input) {
			if err != nil {
				yield(nil, &AskOutput{
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

func runAsk(ctx context.Context, g *genkit.Genkit, def Definition, input *AskInput, cb core.StreamCallback[*AskChunk]) (*AskOutput, error) {
	messages := assembleMessages(def.SystemPrompt, input)

	streamedAny := false
	stream := func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
		text := chunk.Text()
		if text == "" {
			return nil
		}
		if err := cb(ctx, &AskChunk{AnswerDelta: text}); err != nil {
			return err
		}
		streamedAny = true
		return nil
	}

	var resp *ai.ModelResponse
	var err error
	for attempt := 0; ; attempt++ {
		resp, err = genkit.Generate(ctx, g,
			ai.WithMessages(messages...),
			ai.WithTools(def.Tools...),
			ai.WithMaxTurns(askMaxTurns),
			ai.WithStreaming(stream),
		)
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

	out := &AskOutput{
		SessionID:        input.SessionID,
		Answer:           resp.Text(),
		FinishReason:     string(resp.FinishReason),
		ToolCalls:        extractToolCalls(resp),
		PendingToolCalls: extractPendingToolCalls(resp),
	}
	if resp.Usage != nil {
		out.Usage = TokenUsage{
			InputTokens:  resp.Usage.InputTokens,
			OutputTokens: resp.Usage.OutputTokens,
			TotalTokens:  resp.Usage.TotalTokens,
		}
	}
	return out, nil
}

func assembleMessages(systemPrompt string, input *AskInput) []*ai.Message {
	messages := []*ai.Message{ai.NewSystemTextMessage(systemPrompt)}
	for _, m := range input.History {
		if m.Role == "model" {
			messages = append(messages, ai.NewModelTextMessage(m.Text))
			continue
		}
		messages = append(messages, ai.NewUserTextMessage(m.Text))
	}
	return append(messages, ai.NewUserTextMessage(input.UserMessage))
}

func extractToolCalls(resp *ai.ModelResponse) []ToolCall {
	if resp == nil || resp.Request == nil {
		return nil
	}
	var calls []ToolCall
	for _, m := range resp.Request.Messages {
		for _, p := range m.Content {
			if p.IsToolRequest() {
				calls = append(calls, ToolCall{Name: p.ToolRequest.Name, Input: p.ToolRequest.Input})
			}
		}
	}
	return calls
}

// extractPendingToolCalls は履歴中のツール応答から承認待ち（confirmation_required）を拾い上げる。
func extractPendingToolCalls(resp *ai.ModelResponse) []PendingToolCall {
	if resp == nil || resp.Request == nil {
		return nil
	}
	var pendings []PendingToolCall
	for _, m := range resp.Request.Messages {
		for _, p := range m.Content {
			if !p.IsToolResponse() {
				continue
			}
			out, ok := p.ToolResponse.Output.(map[string]any)
			if !ok || out["status"] != "confirmation_required" {
				continue
			}
			id, _ := out["toolCallId"].(string)
			input, _ := out["input"].(map[string]any)
			pendings = append(pendings, PendingToolCall{ID: id, Name: p.ToolResponse.Name, Input: input})
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
