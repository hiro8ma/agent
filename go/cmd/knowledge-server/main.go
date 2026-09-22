// Package main は組織で共有する文書を検索する KnowledgeService の起動口。
//
// 環境変数
//
//	KNOWLEDGE_PORT  待ち受けるポート（既定 19930）
package main

import (
	"context"
	"log/slog"
	"net/http"
	"os"

	"github.com/hiro8ma/agent/go/internal/genkitagent/knowledge"
	"github.com/hiro8ma/agent/go/internal/knowledge/adapter"
	"github.com/hiro8ma/agent/go/internal/lib/libconnect"
	"github.com/hiro8ma/agent/go/internal/lib/libotel"
	"github.com/hiro8ma/agent/go/internal/lib/libserver"
)

func main() {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))
	if err := run(context.Background(), logger); err != nil {
		logger.Error("server exited", "error", err)
		os.Exit(1)
	}
}

func run(ctx context.Context, logger *slog.Logger) error {
	shutdown, err := libotel.Setup(ctx, "knowledge-server")
	if err != nil {
		return err
	}
	port := os.Getenv("KNOWLEDGE_PORT")
	if port == "" {
		port = "19930"
	}
	mux := http.NewServeMux()
	mux.Handle(adapter.NewHandler(knowledge.NewInMemory(), libconnect.HeaderAuthenticator))
	return libserver.Serve(ctx, logger, ":"+port, mux, shutdown)
}
