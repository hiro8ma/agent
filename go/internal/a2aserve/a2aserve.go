// Package a2aserve はエージェントを A2A で公開する口を、2 要素（mTLS のクライアント証明書と、それに結び付いたトークン）で守って立てる。
//
// 環境変数 A2A_ADDR が無ければ口を出さない。出すときは TLS の証明書、クライアントの CA、IdP の設定がそろっていなければ起動を止める。
// 認証なしで A2A を出す設定は持たない。
package a2aserve

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2asrv"

	"github.com/hiro8ma/agent/go/internal/action"
	"github.com/hiro8ma/agent/go/internal/adk/a2ainterop"
	"github.com/hiro8ma/agent/go/internal/lib/libauth"
	"github.com/hiro8ma/agent/go/internal/toolscope"
)

// Policy は既定のツールごとのスコープ。載っていないツール（MCP で足したものなど）は A2A からは使えない。
var Policy = toolscope.Policy{
	"search_knowledge":             {"knowledge.read"},
	"get_order":                    {"orders.read"},
	"resolve_area_names":           {"orders.read"},
	action.ToolUpdatePaymentMethod: {"orders.write"},
}

// Scopes は Agent Card に載せるスコープと説明。
var Scopes = map[string]string{
	"a2a.invoke":     "エージェントを呼ぶ",
	"knowledge.read": "社内ナレッジを検索する",
	"orders.read":    "注文とエリアを参照する",
	"orders.write":   "注文の変更を申請する",
}

// Config は A2A の口の設定。
type Config struct {
	Addr     string
	BaseURL  string
	AgentID  string
	CertFile string
	KeyFile  string
	ClientCA string
	JWKSURL  string
	IdPCA    string
	Issuer   string
	Audience string
	TokenURL string
	Required []string
	// DevAllowUnboundTokens は証明書に結び付いていないトークンも通す。開発用。mTLS は外さない。
	DevAllowUnboundTokens bool
}

// ConfigFromEnv は環境変数から設定を読む。A2A_ADDR が無ければ ok=false。
func ConfigFromEnv() (cfg Config, ok bool, err error) {
	cfg = Config{
		Addr:                  os.Getenv("A2A_ADDR"),
		BaseURL:               os.Getenv("A2A_BASE_URL"),
		AgentID:               envOr("A2A_AGENT", "operations"),
		CertFile:              os.Getenv("A2A_TLS_CERT"),
		KeyFile:               os.Getenv("A2A_TLS_KEY"),
		ClientCA:              os.Getenv("A2A_CLIENT_CA"),
		JWKSURL:               os.Getenv("A2A_JWKS_URL"),
		IdPCA:                 os.Getenv("A2A_IDP_CA"),
		Issuer:                os.Getenv("A2A_ISSUER"),
		Audience:              os.Getenv("A2A_AUDIENCE"),
		TokenURL:              os.Getenv("A2A_TOKEN_URL"),
		Required:              strings.Fields(envOr("A2A_REQUIRED_SCOPE", "a2a.invoke")),
		DevAllowUnboundTokens: os.Getenv("A2A_DEV_ALLOW_UNBOUND_TOKENS") == "true",
	}
	if cfg.Addr == "" {
		return cfg, false, nil
	}
	var missing []string
	for name, v := range map[string]string{
		"A2A_TLS_CERT": cfg.CertFile, "A2A_TLS_KEY": cfg.KeyFile, "A2A_CLIENT_CA": cfg.ClientCA,
		"A2A_JWKS_URL": cfg.JWKSURL, "A2A_ISSUER": cfg.Issuer, "A2A_AUDIENCE": cfg.Audience, "A2A_TOKEN_URL": cfg.TokenURL,
	} {
		if v == "" {
			missing = append(missing, name)
		}
	}
	if len(missing) > 0 {
		return cfg, true, fmt.Errorf("a2aserve: A2A_ADDR を設定したときは %s も必要", strings.Join(missing, ", "))
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://" + cfg.Addr
	}
	return cfg, true, nil
}

// Card は 2 要素を 1 つの要件（AND）として宣言した Agent Card を返す。
func Card(cfg Config, name, description string) a2a.AgentCard {
	card := a2a.AgentCard{
		Name: name, Description: description, Version: "1.0.0",
		DefaultInputModes: []string{"text/plain"}, DefaultOutputModes: []string{"text/plain"},
		Skills: []a2a.AgentSkill{{ID: name, Name: name, Description: description, Tags: []string{"agent"}}},
	}
	a2ainterop.DeclareCertBoundOAuth2(&card, "oauth2", "mtls", cfg.TokenURL, Scopes, cfg.Required...)
	return card
}

// Protect は Agent Card を除く口を、Verifier で守る。
func Protect(h http.Handler, v *libauth.Verifier, cfg Config) http.Handler {
	return libauth.Middleware(h, v, []string{a2asrv.WellKnownAgentCardPath}, cfg.Required...)
}

// Verifier は設定から Verifier を作る。
func Verifier(ctx context.Context, cfg Config) (*libauth.Verifier, error) {
	var client *http.Client
	if cfg.IdPCA != "" {
		pem, err := os.ReadFile(filepath.Clean(cfg.IdPCA))
		if err != nil {
			return nil, fmt.Errorf("a2aserve: idp ca: %w", err)
		}
		pool := x509.NewCertPool()
		if !pool.AppendCertsFromPEM(pem) {
			return nil, errors.New("a2aserve: no certificate in A2A_IDP_CA")
		}
		client = &http.Client{Timeout: 10 * time.Second, Transport: &http.Transport{TLSClientConfig: &tls.Config{RootCAs: pool, MinVersion: tls.VersionTLS12}}}
	}
	return libauth.NewVerifier(ctx, libauth.Config{
		JWKSURL: cfg.JWKSURL, Issuer: cfg.Issuer, Audience: cfg.Audience,
		RequireCertBinding: !cfg.DevAllowUnboundTokens, HTTPClient: client, Leeway: 30 * time.Second,
	})
}

// Serve はクライアント証明書を必須にした TLS で h を公開し、ctx が終わると止める。
func Serve(ctx context.Context, logger *slog.Logger, cfg Config, h http.Handler) error {
	tlsCfg, err := libauth.ServerTLSConfig(cfg.CertFile, cfg.KeyFile, cfg.ClientCA)
	if err != nil {
		return err
	}
	srv := &http.Server{Addr: cfg.Addr, Handler: h, TLSConfig: tlsCfg, ReadHeaderTimeout: 10 * time.Second}
	errc := make(chan error, 1)
	go func() { errc <- srv.ListenAndServeTLS("", "") }()
	logger.InfoContext(ctx, "a2a listening", "addr", cfg.Addr, "agent", cfg.AgentID, "unbound_tokens_allowed", cfg.DevAllowUnboundTokens)
	select {
	case err := <-errc:
		return err
	case <-ctx.Done():
		shutdownCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 10*time.Second)
		defer cancel()
		return srv.Shutdown(shutdownCtx)
	}
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
