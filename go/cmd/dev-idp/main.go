// dev-idp は手元のデモのための認証局と IdP。本番では使わない。
//
// 起動のたびに認証局と証明書を作り直して -dir に書き、client_credentials のトークンを mTLS で発行する。
// トークンはクライアント証明書に結び付く（cnf.x5t#S256）。
//
//	go run ./cmd/dev-idp -dir ../ops/.run/pki -server adk-agent -server genkit-agent \
//	    -client genkit-agent=a2a.invoke,orders.read -client adk-agent=a2a.invoke,orders.read
package main

import (
	"context"
	"crypto/tls"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/hiro8ma/agent/go/internal/devidp"
)

type multi []string

func (m *multi) String() string     { return strings.Join(*m, ",") }
func (m *multi) Set(v string) error { *m = append(*m, v); return nil }

func main() {
	addr := flag.String("addr", "127.0.0.1:19800", "待ち受けるアドレス")
	dir := flag.String("dir", "../ops/.run/pki", "証明書と鍵を書くディレクトリ（git に入れない場所）")
	audience := flag.String("audience", "a2a://agents", "発行するトークンの aud")
	var clients, servers multi
	flag.Var(&clients, "client", "クライアント証明書の CN と許可するスコープ（name=scope1,scope2）。繰り返せる")
	flag.Var(&servers, "server", "サーバー証明書を発行する名前。繰り返せる")
	flag.Parse()
	if err := run(*addr, *dir, *audience, clients, servers); err != nil {
		slog.Error("dev-idp", "error", err)
		os.Exit(1)
	}
}

func run(addr, dir, audience string, clients, servers []string) error {
	if len(clients) == 0 {
		return errors.New("-client を 1 つ以上指定する")
	}
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}
	ca, err := devidp.NewCA("dev-ca")
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, "ca.pem"), ca.PEM(), 0o600); err != nil {
		return err
	}
	scopes := map[string][]string{}
	for _, c := range clients {
		name, list, ok := strings.Cut(c, "=")
		if !ok || name == "" {
			return fmt.Errorf("-client は name=scope1,scope2 の形: %q", c)
		}
		scopes[name] = strings.Split(list, ",")
		issued, err := ca.IssueClient(name)
		if err != nil {
			return err
		}
		if err := write(dir, name+"-client", issued); err != nil {
			return err
		}
	}
	for _, name := range append([]string{"idp"}, servers...) {
		issued, err := ca.IssueServer(name)
		if err != nil {
			return err
		}
		if err := write(dir, name+"-server", issued); err != nil {
			return err
		}
	}
	idpCert, err := tls.LoadX509KeyPair(filepath.Join(dir, "idp-server.pem"), filepath.Join(dir, "idp-server.key"))
	if err != nil {
		return err
	}
	issuer := "https://" + addr
	idp, err := devidp.New(issuer, audience, scopes)
	if err != nil {
		return err
	}
	srv := &http.Server{
		Addr: addr, Handler: idp.Handler(), ReadHeaderTimeout: 5 * time.Second,
		TLSConfig: &tls.Config{Certificates: []tls.Certificate{idpCert}, ClientCAs: ca.Pool(), ClientAuth: tls.VerifyClientCertIfGiven, MinVersion: tls.VersionTLS13},
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	go func() {
		<-ctx.Done()
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = srv.Shutdown(shutdownCtx)
	}()
	slog.Info("dev-idp listening", "issuer", issuer, "jwks", issuer+devidp.JWKSPath, "token", issuer+devidp.TokenPath, "dir", dir)
	if err := srv.ListenAndServeTLS("", ""); !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	return nil
}

func write(dir, name string, issued *devidp.Issued) error {
	if err := os.WriteFile(filepath.Join(dir, name+".pem"), issued.CertPEM, 0o600); err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, name+".key"), issued.KeyPEM, 0o600)
}
