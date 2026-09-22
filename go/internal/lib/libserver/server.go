// Package libserver は Connect のサービスを HTTP で起動する共通処理。
package libserver

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"os/signal"
	"syscall"
	"time"
)

// Serve は mux を addr で公開し、SIGINT / SIGTERM で処理中のリクエストを待ってから止まる。
// TLS なしの HTTP/2 も受けるので、Connect の server streaming をそのまま通せる。
// onShutdown はリクエストを捌き終えた後に呼ぶ（telemetry の送り切りなど）。
func Serve(ctx context.Context, logger *slog.Logger, addr string, mux http.Handler, onShutdown ...func(context.Context) error) error {
	protocols := new(http.Protocols)
	protocols.SetHTTP1(true)
	protocols.SetUnencryptedHTTP2(true)
	// streaming は長く続くので WriteTimeout は置かず、ヘッダの読み取りだけを区切る。
	srv := &http.Server{
		Addr:              addr,
		Handler:           mux,
		Protocols:         protocols,
		ReadHeaderTimeout: 10 * time.Second,
	}

	ctx, stop := signal.NotifyContext(ctx, syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	errc := make(chan error, 1)
	go func() { errc <- srv.ListenAndServe() }()
	logger.Info("listening", "addr", addr)

	select {
	case err := <-errc:
		return err
	case <-ctx.Done():
	}
	shutdownCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 10*time.Second)
	defer cancel()
	errs := []error{srv.Shutdown(shutdownCtx)}
	if err := <-errc; !errors.Is(err, http.ErrServerClosed) {
		errs = append(errs, err)
	}
	for _, f := range onShutdown {
		errs = append(errs, f(shutdownCtx))
	}
	return errors.Join(errs...)
}
