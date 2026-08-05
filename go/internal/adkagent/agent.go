// Package adkagent は AgentService の ADK（google.golang.org/adk/v2）実装。
// genkit 版（internal/genkitagent）と同じ agentcore.Agent を満たし、transport を共有する。
package adkagent

import (
	"context"
	"iter"
	"strings"

	adkagentpkg "google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/genai"

	"github.com/hiro8ma/agent/go/internal/agentcore"
)

const defaultUser = "local-user"

// Definition はエージェント 1 体の定義。genkit 版の Definition と対になる。
type Definition struct {
	ID          string
	Description string
	Instruction string
	Model       model.LLM
	Tools       []tool.Tool
}

// Agent は 1 つの Definition を ADK Runner として公開する。
// セッション履歴は ADK の SessionService が持つため、AskInput.History は使わない
// （transport が渡してくるが、同一プロセス内は ADK 側の履歴が正になる）。
type Agent struct {
	def    Definition
	runner *runner.Runner
}

func New(def Definition) (*Agent, error) {
	ag, err := llmagent.New(llmagent.Config{
		Name:        def.ID,
		Description: def.Description,
		Instruction: def.Instruction,
		Model:       def.Model,
		Tools:       def.Tools,
	})
	if err != nil {
		return nil, err
	}
	r, err := runner.New(runner.Config{
		AppName:           "adk-agent-" + def.ID,
		Agent:             ag,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
	})
	if err != nil {
		return nil, err
	}
	return &Agent{def: def, runner: r}, nil
}

// Info は agentcore.Agent の実装。
func (a *Agent) Info() agentcore.AgentInfo {
	return agentcore.AgentInfo{ID: a.def.ID, Description: a.def.Description}
}

var _ agentcore.Agent = (*Agent)(nil)

// Ask は Runner のイベント列を agentcore のチャンク / 最終出力に写像する。
// partial イベント → AnswerDelta、非 partial のテキスト → 最終 Answer、
// FunctionCall パート → ToolCalls。エラーも AskOutput.ErrorMessage に畳み込む。
func (a *Agent) Ask(ctx context.Context, input *agentcore.AskInput) iter.Seq2[*agentcore.AskChunk, *agentcore.AskOutput] {
	return func(yield func(*agentcore.AskChunk, *agentcore.AskOutput) bool) {
		out := &agentcore.AskOutput{SessionID: input.SessionID}
		msg := genai.NewContentFromText(input.UserMessage, genai.RoleUser)
		cfg := adkagentpkg.RunConfig{StreamingMode: adkagentpkg.StreamingModeSSE}

		var finalText strings.Builder
		for event, err := range a.runner.Run(ctx, defaultUser, input.SessionID, msg, cfg) {
			if err != nil {
				out.FinishReason = "error"
				out.ErrorMessage = err.Error()
				yield(nil, out)
				return
			}
			if event == nil {
				continue
			}
			if !event.Partial && event.UsageMetadata != nil {
				out.Usage.InputTokens += int(event.UsageMetadata.PromptTokenCount)
				out.Usage.OutputTokens += int(event.UsageMetadata.CandidatesTokenCount)
				out.Usage.TotalTokens += int(event.UsageMetadata.TotalTokenCount)
			}
			if event.FinishReason != "" {
				out.FinishReason = strings.ToLower(string(event.FinishReason))
			}
			if event.Content == nil {
				continue
			}
			for _, part := range event.Content.Parts {
				// FunctionCall は partial と非 partial の両イベントに現れるため、非 partial だけ記録する。
				if part.FunctionCall != nil && !event.Partial {
					out.ToolCalls = append(out.ToolCalls, agentcore.ToolCall{
						Name:  part.FunctionCall.Name,
						Input: part.FunctionCall.Args,
					})
				}
				if part.Text == "" {
					continue
				}
				if event.Partial {
					if !yield(&agentcore.AskChunk{AnswerDelta: part.Text}, nil) {
						return
					}
					continue
				}
				// 非 partial イベントは 1 ターン分の全文を持つ。最後のテキストターンが最終回答。
				finalText.Reset()
				finalText.WriteString(part.Text)
			}
		}

		out.Answer = strings.TrimSpace(finalText.String())
		if out.FinishReason == "" {
			out.FinishReason = "stop"
		}
		yield(nil, out)
	}
}
