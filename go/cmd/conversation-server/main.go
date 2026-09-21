// Package main は会話の履歴を持つ ConversationService の起動口。
//
// 環境変数
//
//	CONVERSATION_PORT  待ち受けるポート（既定 19920）
//	CONVERSATION_DSN   SQLite の保管先（既定 file:./.conversation/conversation.db）
package main

import (
	"context"
	"log/slog"
	"net/http"
	"os"

	"github.com/hiro8ma/agent/go/internal/conversation/adapter"
	"github.com/hiro8ma/agent/go/internal/conversation/repository"
	"github.com/hiro8ma/agent/go/internal/conversation/usecase"
	"github.com/hiro8ma/agent/go/internal/lib/libconnect"
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
	repo, err := repository.NewSQLite(envOr("CONVERSATION_DSN", "file:./.conversation/conversation.db"))
	if err != nil {
		return err
	}
	mux := http.NewServeMux()
	mux.Handle(adapter.NewHandler(usecase.New(repo), libconnect.HeaderAuthenticator))
	return libserver.Serve(ctx, logger, ":"+envOr("CONVERSATION_PORT", "19920"), mux)
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
