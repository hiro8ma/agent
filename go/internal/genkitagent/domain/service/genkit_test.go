package service

import (
	"context"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"

	"github.com/hiro8ma/agent/go/internal/genkitagent/domain/model"
)

func newEchoAgent(t *testing.T) *GenkitAgent {
	t.Helper()
	g := genkit.Init(t.Context(), genkit.WithDefaultModel("test/echo"))
	genkit.DefineModel(g, "test/echo", &ai.ModelOptions{Supports: &ai.ModelSupports{Multiturn: true, SystemRole: true}},
		func(_ context.Context, req *ai.ModelRequest, _ ai.ModelStreamCallback) (*ai.ModelResponse, error) {
			last := req.Messages[len(req.Messages)-1]
			return &ai.ModelResponse{
				Message:      ai.NewModelTextMessage("echo: " + last.Text()),
				FinishReason: ai.FinishReasonStop,
			}, nil
		})
	return NewGenkitAgent(g, Definition{ID: "echo", SystemPrompt: "テスト"})
}

func TestChatAcceptsNilHistory(t *testing.T) {
	t.Parallel()
	a := newEchoAgent(t)

	testCases := map[string]struct {
		history []model.Message
	}{
		"新しいセッションの履歴は nil": {history: nil},
		"空のスライス":           {history: []model.Message{}},
		"履歴あり":             {history: []model.Message{{Role: "user", Text: "前の質問"}, {Role: "model", Text: "前の回答"}}},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			var final *model.ChatOutput
			for _, out := range a.Chat(t.Context(), &model.ChatInput{SessionID: "s1", UserMessage: "こんにちは", History: tc.history}) {
				if out != nil {
					final = out
				}
			}
			if final == nil || final.ErrorMessage != "" {
				t.Fatalf("Chat() = %+v", final)
			}
			if final.Answer != "echo: こんにちは" {
				t.Errorf("Answer = %q", final.Answer)
			}
		})
	}
}
