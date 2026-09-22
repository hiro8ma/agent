// Package main は変更系の操作の承認を扱う ActionService の起動口。
//
// 環境変数
//
//	ACTION_PORT       待ち受けるポート（既定 19940）
//	ACTION_APPROVERS  高リスクの操作を承認できる利用者（カンマ区切り）
package main

import (
	"context"
	"log/slog"
	"net/http"
	"os"

	"github.com/hiro8ma/agent/go/internal/action"
	"github.com/hiro8ma/agent/go/internal/action/adapter"
	"github.com/hiro8ma/agent/go/internal/lib/libconnect"
	"github.com/hiro8ma/agent/go/internal/lib/liblog"
	"github.com/hiro8ma/agent/go/internal/lib/libotel"
	"github.com/hiro8ma/agent/go/internal/lib/libserver"
)

func main() {
	logger, err := liblog.InitFromEnv("action-server")
	if err != nil {
		slog.Error("ロガーを作れない", "error", err)
		os.Exit(1)
	}
	if err := run(context.Background(), logger); err != nil {
		logger.Error("server exited", "error", err)
		os.Exit(1)
	}
}

func run(ctx context.Context, logger *slog.Logger) error {
	shutdown, err := libotel.Setup(ctx, "action-server")
	if err != nil {
		return err
	}
	port := os.Getenv("ACTION_PORT")
	if port == "" {
		port = "19940"
	}
	local := action.NewLocalFromEnv()
	mux := http.NewServeMux()
	mux.Handle(adapter.NewHandler(local.Service, libconnect.HeaderAuthenticator))
	return libserver.Serve(ctx, logger, ":"+port, mux, shutdown)
}
