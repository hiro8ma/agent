// Package main は A2A の相互運用を確かめるための、決まった文を返すエージェントのサーバー。
//
// モデルを呼ばない。Python の ADK（a2a-sdk 0.3）の RemoteA2aAgent から、Go の ADK の A2A v1.0 のサーバーを
// 呼べるかを確かめるテストが起動する。
//
//	go run ./cmd/a2a-interop-server -addr 127.0.0.1:18765
package main

import (
	"context"
	"flag"
	"iter"
	"log"
	"net/http"
	"time"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/genai"

	"github.com/hiro8ma/agent/go/internal/adk/a2ainterop"
)

type echo struct{}

func (echo) Name() string { return "echo" }

func (echo) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	text := req.Contents[len(req.Contents)-1].Parts[0].Text
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: &genai.Content{Role: "model", Parts: []*genai.Part{{Text: "Go で受け付けました: " + text}}}}, nil)
	}
}

func main() {
	addr := flag.String("addr", "127.0.0.1:18765", "待ち受けるアドレス")
	flag.Parse()
	a, err := llmagent.New(llmagent.Config{Name: "go_expense_agent", Model: echo{}, Instruction: "x"})
	if err != nil {
		log.Fatal(err)
	}
	card := a2a.AgentCard{
		Name: "go-expense-agent", Description: "Go の ADK で動く経費のエージェント", Version: "1.0.0",
		DefaultInputModes: []string{"text/plain"}, DefaultOutputModes: []string{"text/plain"},
		Skills: []a2a.AgentSkill{{ID: "expense", Name: "経費", Description: "経費の受付", Tags: []string{"expense"}}},
	}
	srv := &http.Server{Addr: *addr, Handler: a2ainterop.NewHandler(a, card, "http://"+*addr), ReadHeaderTimeout: 5 * time.Second}
	log.Fatal(srv.ListenAndServe())
}
