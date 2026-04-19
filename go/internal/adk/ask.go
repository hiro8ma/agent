package adk

import (
	"context"
	"strings"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/genai"

	"github.com/hiro8ma/agent/go/internal/agent/domain/model"
	"github.com/hiro8ma/agent/go/internal/agent/usecase"
	"github.com/hiro8ma/agent/go/internal/lib/liberrors"
)

const (
	appName     = "agent_adk"
	defaultUser = "local_user"
)

// Ask は Google ADK Runner を用いた usecase.Ask 実装。
// Runner が Tool Dispatch ループを内包するため、ここにループは書かない。
type Ask struct {
	runner    *runner.Runner
	sessionID string
}

// Config は Ask 初期化パラメータ。
type Config struct {
	APIKey       string
	ModelName    string
	SystemPrompt string
	AgentName    string
}

// New は Ask 実装を生成する。
func New(ctx context.Context, cfg Config) (*Ask, error) {
	m, err := gemini.NewModel(ctx, cfg.ModelName, &genai.ClientConfig{
		APIKey: cfg.APIKey,
	})
	if err != nil {
		return nil, liberrors.Wrap(liberrors.CodeUnavailable, err, "gemini model init")
	}

	tools, err := buildTools()
	if err != nil {
		return nil, liberrors.Wrap(liberrors.CodeInternal, err, "build tools")
	}

	ag, err := llmagent.New(llmagent.Config{
		Name:        cfg.AgentName,
		Description: "comparison agent (adk)",
		Instruction: cfg.SystemPrompt,
		Model:       m,
		Tools:       tools,
	})
	if err != nil {
		return nil, liberrors.Wrap(liberrors.CodeInternal, err, "llmagent new")
	}

	sessionSvc := session.InMemoryService()
	r, err := runner.New(runner.Config{
		AppName:        appName,
		Agent:          ag,
		SessionService: sessionSvc,
	})
	if err != nil {
		return nil, liberrors.Wrap(liberrors.CodeInternal, err, "runner new")
	}

	// 1 デモプロセス 1 セッション。ADK が自動生成する ID を受け取るため空文字で初期化
	createResp, err := sessionSvc.Create(ctx, &session.CreateRequest{
		AppName: appName,
		UserID:  defaultUser,
	})
	if err != nil {
		return nil, liberrors.Wrap(liberrors.CodeInternal, err, "session create")
	}

	return &Ask{runner: r, sessionID: createResp.Session.ID()}, nil
}

// Handle は 1 ターンの user message を Runner に流し、最終応答を収集する。
func (a *Ask) Handle(ctx context.Context, in usecase.AskInput) (usecase.AskOutput, error) {
	conv := in.Conversation
	if conv == nil {
		conv = &model.Conversation{}
	}
	conv.Append(model.Message{Role: model.RoleUser, Text: in.UserMessage})

	userMsg := genai.NewContentFromText(in.UserMessage, genai.RoleUser)

	var textChunks []string
	totalTokens := 0
	for event, err := range a.runner.Run(ctx, defaultUser, a.sessionID, userMsg, agent.RunConfig{}) {
		if err != nil {
			return usecase.AskOutput{}, liberrors.Wrap(liberrors.CodeUnavailable, err, "runner run")
		}
		if event == nil || event.Content == nil {
			continue
		}
		for _, part := range event.Content.Parts {
			if part.Text != "" {
				textChunks = append(textChunks, part.Text)
			}
		}
		if event.UsageMetadata != nil {
			totalTokens += int(event.UsageMetadata.TotalTokenCount)
		}
	}

	answer := strings.TrimSpace(strings.Join(textChunks, ""))
	conv.Append(model.Message{Role: model.RoleAssistant, Text: answer})

	return usecase.AskOutput{
		Answer:       answer,
		Conversation: conv,
		Iterations:   1, // Runner が内部でループ。外からは 1 回に見える
		TotalTokens:  totalTokens,
	}, nil
}
