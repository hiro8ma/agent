package guardrail_test

import (
	"context"
	"errors"
	"iter"
	"strings"
	"testing"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/genai"

	"github.com/hiro8ma/agent/go/internal/guardrail"
)

// failingModel は無料枠の上限に当たったときのように、毎回失敗する。
type failingModel struct{}

func (failingModel) Name() string { return "failing" }

func (failingModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(nil, errors.New("429 RESOURCE_EXHAUSTED quota exceeded for project p-123"))
	}
}

func TestFallbackMarksTheReplacedResponse(t *testing.T) {
	t.Parallel()
	a, err := llmagent.New(llmagent.Config{
		Name:  "weather_agent",
		Model: failingModel{},
		OnModelErrorCallbacks: []llmagent.OnModelErrorCallback{
			guardrail.FallbackOnModelError(guardrail.NewLog(), "いま天気を取得できません。"),
		},
	})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	r, err := runner.New(runner.Config{AppName: "weather", Agent: a, SessionService: session.InMemoryService(), AutoCreateSession: true})
	if err != nil {
		t.Fatalf("runner.New() error = %v", err)
	}

	var last *session.Event
	msg := &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "東京の天気は？"}}}
	for ev, err := range r.Run(t.Context(), "u1", "s1", msg, agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
		last = ev
	}

	if last == nil || last.Content == nil || last.Content.Parts[0].Text != "いま天気を取得できません。" {
		t.Fatalf("利用者向けの決まった文が返らない: %+v", last)
	}
	if last.ErrorCode != guardrail.ModelErrorFallback {
		t.Errorf("ErrorCode = %q, want %q", last.ErrorCode, guardrail.ModelErrorFallback)
	}
	if last.ErrorMessage == "" || strings.Contains(last.ErrorMessage, "p-123") {
		t.Errorf("ErrorMessage = %q。空か、元のエラーの文言を含む", last.ErrorMessage)
	}
}
