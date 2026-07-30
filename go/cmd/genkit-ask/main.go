// ask は動作確認用の CLI クライアント。
//
//	go run ./cmd/ask -agent operations "注文 ord-001 の支払い方法を教えて"
//	go run ./cmd/ask -list
//	go run ./cmd/ask -exec <toolCallId>
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"time"

	"connectrpc.com/connect"

	agentv1 "github.com/hiro8ma/agent/go/gen/agent/v1"
	"github.com/hiro8ma/agent/go/gen/agent/v1/agentv1connect"
)

func main() {
	var (
		baseURL    = flag.String("url", "http://localhost:19910", "サーバー URL")
		agentID    = flag.String("agent", "operations", "エージェント ID")
		sessionID  = flag.String("session", "local-session", "セッション ID")
		list       = flag.Bool("list", false, "エージェント一覧を表示")
		execCallID = flag.String("exec", "", "承認済みツール呼び出しを実行する toolCallId")
	)
	flag.Parse()

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	client := agentv1connect.NewAgentServiceClient(http.DefaultClient, *baseURL)

	switch {
	case *list:
		resp, err := client.ListAgents(ctx, connect.NewRequest(&agentv1.ListAgentsRequest{}))
		if err != nil {
			log.Fatalf("list agents: %v", err)
		}
		for _, a := range resp.Msg.GetAgents() {
			fmt.Printf("%s\t%s\n", a.GetId(), a.GetDescription())
		}
	case *execCallID != "":
		resp, err := client.ExecuteConfirmedToolCall(ctx, connect.NewRequest(&agentv1.ExecuteConfirmedToolCallRequest{
			ToolCallId: *execCallID,
		}))
		if err != nil {
			log.Fatalf("execute confirmed tool call: %v", err)
		}
		fmt.Printf("executed: %v\n", resp.Msg.GetResult().AsMap())
	default:
		if flag.NArg() == 0 {
			log.Fatal("message is required")
		}
		ask(ctx, client, *agentID, *sessionID, flag.Arg(0))
	}
}

func ask(ctx context.Context, client agentv1connect.AgentServiceClient, agentID, sessionID, message string) {
	stream, err := client.Ask(ctx, connect.NewRequest(&agentv1.AskRequest{
		AgentId:   agentID,
		SessionId: sessionID,
		Message:   message,
	}))
	if err != nil {
		log.Fatalf("ask: %v", err)
	}
	for stream.Receive() {
		msg := stream.Msg()
		if delta := msg.GetAnswerDelta(); delta != "" {
			fmt.Print(delta)
			continue
		}
		if result := msg.GetResult(); result != nil {
			fmt.Println()
			printResult(result)
		}
	}
	if err := stream.Err(); err != nil {
		log.Fatalf("stream: %v", err)
	}
}

func printResult(result *agentv1.AskResult) {
	if result.GetErrorMessage() != "" {
		fmt.Fprintf(os.Stderr, "error: %s\n", result.GetErrorMessage())
		return
	}
	for _, tc := range result.GetToolCalls() {
		fmt.Printf("[tool] %s %v\n", tc.GetName(), tc.GetInput().AsMap())
	}
	for _, p := range result.GetPendingToolCalls() {
		fmt.Printf("[pending] %s %v\n  承認して実行: go run ./cmd/ask -exec %s\n", p.GetName(), p.GetInput().AsMap(), p.GetToolCallId())
	}
	usage := result.GetUsage()
	fmt.Printf("[usage] in=%d out=%d finish=%s\n", usage.GetInputTokens(), usage.GetOutputTokens(), result.GetFinishReason())
}
