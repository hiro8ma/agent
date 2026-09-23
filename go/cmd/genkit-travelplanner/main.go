// Package main は旅行プランナーの Genkit 版のエントリ。
//
// フローを HTTP で公開し、プロセスを待機させる。
// GENKIT_ENV=dev で起動すると Developer UI からもフローを実行できる（genkit start -- go run ./cmd/genkit-travelplanner）。
package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"

	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"github.com/firebase/genkit/go/plugins/server"

	"github.com/hiro8ma/agent/go/internal/genkit/travelplanner"
)

func main() {
	ctx := context.Background()

	if os.Getenv("GEMINI_API_KEY") == "" && os.Getenv("GOOGLE_API_KEY") == "" {
		log.Fatal("GEMINI_API_KEY または GOOGLE_API_KEY を設定してください")
	}

	g := genkit.Init(ctx,
		genkit.WithPlugins(&googlegenai.GoogleAI{}),
		genkit.WithDefaultModel("googleai/"+travelplanner.ModelName),
	)
	flow := travelplanner.DefineFlow(g)

	port := 3400
	if v := os.Getenv("PORT"); v != "" {
		p, err := strconv.Atoi(v)
		if err != nil {
			log.Fatalf("PORT は数値で指定してください: %v", err)
		}
		port = p
	}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /"+flow.Name(), genkit.Handler(flow))

	addr := fmt.Sprintf("127.0.0.1:%d", port)
	log.Printf("flow: POST http://%s/%s", addr, flow.Name())
	if err := server.Start(ctx, addr, mux); err != nil {
		log.Fatal(err)
	}
}
