package cligate_test

import (
	"context"
	"iter"
	"os"
	"testing"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/genai"
)

// callArgs は 1 回目に決めた引数で run_gcloud を呼び、応答を受けたら答える。
type callArgs []any

func (c callArgs) Name() string { return "scripted" }

func (c callArgs) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	last := req.Contents[len(req.Contents)-1]
	part := &genai.Part{Text: "終わりました"}
	if last.Parts[0].FunctionResponse == nil {
		part = &genai.Part{FunctionCall: &genai.FunctionCall{Name: "run_gcloud", Args: map[string]any{"args": []any(c)}}}
	}
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: &genai.Content{Role: "model", Parts: []*genai.Part{part}}}, nil)
	}
}

func TestToolReturnsRejectionToTheModel(t *testing.T) {
	t.Parallel()
	testCases := map[string]struct {
		args       callArgs
		wantStatus string
		wantRun    bool
	}{
		"許可したコマンドは実行する": {args: callArgs{"storage", "buckets", "list", "--project=p"}, wantStatus: "ok", wantRun: true},
		"トークンの表示は理由を返す": {args: callArgs{"auth", "print-access-token"}, wantStatus: "rejected"},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			bin, marker := fakeGcloud(t, "")
			gt, err := gate(bin).Tool("run_gcloud", "gcloud の許可したコマンドを実行する")
			if err != nil {
				t.Fatal(err)
			}
			a, err := llmagent.New(llmagent.Config{Name: "ops", Model: tc.args, Tools: []tool.Tool{gt}})
			if err != nil {
				t.Fatal(err)
			}
			r, err := runner.New(runner.Config{AppName: "ops", Agent: a, SessionService: session.InMemoryService(), AutoCreateSession: true})
			if err != nil {
				t.Fatal(err)
			}
			var status any
			msg := &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "一覧を出して"}}}
			for ev, err := range r.Run(t.Context(), "u1", "s1", msg, agent.RunConfig{}) {
				if err != nil {
					t.Fatal(err)
				}
				if ev.Content != nil && ev.Content.Parts[0].FunctionResponse != nil {
					status = ev.Content.Parts[0].FunctionResponse.Response["status"]
				}
			}
			if status != tc.wantStatus {
				t.Errorf("status = %v, want %v", status, tc.wantStatus)
			}
			if _, err := os.Stat(marker); (err == nil) != tc.wantRun {
				t.Errorf("実行された = %v, want %v", err == nil, tc.wantRun)
			}
		})
	}
}
