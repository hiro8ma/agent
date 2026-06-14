package workflow

import (
	"context"
	"testing"

	"github.com/hiro8ma/agent/go/internal/agent/usecase"
)

// fakeAsk は usecase.Ask の決定論スタブ（LLM 不要）。
type fakeAsk struct{ answer string }

func (f fakeAsk) Handle(_ context.Context, in usecase.AskInput) (usecase.AskOutput, error) {
	return usecase.AskOutput{Answer: f.answer + ":" + in.UserMessage, Iterations: 1, TotalTokens: 7}, nil
}

func TestLLMActivity_WrapsAsk(t *testing.T) {
	act := LLMActivity{ActivityName: "triage", Ask: fakeAsk{answer: "ok"}}
	res, err := act.Execute(context.Background(), ActivityInput{Messages: []string{"hello"}})
	if err != nil {
		t.Fatalf("execute: %v", err)
	}
	if got := res.Output["answer"]; got != "ok:hello" {
		t.Fatalf("unexpected answer: %v", got)
	}
}

// LLMActivity が Activity を満たすことをコンパイル時に保証する。
var _ Activity = LLMActivity{}
