package libauth

import (
	"context"
	"crypto/subtle"
	"crypto/x509"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/MicahParks/keyfunc/v3"
	"github.com/golang-jwt/jwt/v5"
	"golang.org/x/time/rate"
)

var (
	// ErrInvalidToken はトークンの署名、発行者、宛先、期限のどれかが正しくない。
	ErrInvalidToken = errors.New("libauth: invalid token")
	// ErrCertBinding はトークンが提示された証明書に結び付いていない。
	ErrCertBinding = errors.New("libauth: token is not bound to the client certificate")
)

// Config は Verifier の設定。
type Config struct {
	JWKSURL  string
	Issuer   string
	Audience string
	// Algorithms は受け付ける署名の方式。既定は ES256 と RS256。none と HS 系は指定しても受け付けない。
	Algorithms []string
	Leeway     time.Duration
	// RequireCertBinding はトークンの cnf.x5t#S256 と、提示された証明書の一致を求める（RFC 8705）。
	RequireCertBinding bool
	// HTTPClient は JWKS を取るクライアント。IdP を私設の CA で立てるときに渡す。
	HTTPClient *http.Client
	// RefreshUnknownKID は知らない kid のときに JWKS を取り直す頻度の上限。既定は 1 分に 1 回。
	RefreshUnknownKID *rate.Limiter
}

// Verifier は JWT を JWKS で検証する。JWKS はキャッシュし、知らない kid のときだけ取り直す。
type Verifier struct {
	cfg    Config
	keys   keyfunc.Keyfunc
	parser *jwt.Parser
}

var defaultAlgorithms = []string{"ES256", "RS256"}

// NewVerifier は JWKS の取得を始め、Verifier を返す。ctx が終わると JWKS の定期の更新も止まる。
func NewVerifier(ctx context.Context, cfg Config) (*Verifier, error) {
	if cfg.JWKSURL == "" || cfg.Issuer == "" || cfg.Audience == "" {
		return nil, errors.New("libauth: JWKSURL, Issuer, Audience are required")
	}
	algs := cfg.Algorithms
	if len(algs) == 0 {
		algs = defaultAlgorithms
	}
	for _, a := range algs {
		if a == "none" || strings.HasPrefix(a, "HS") {
			return nil, fmt.Errorf("libauth: algorithm %q is not allowed", a)
		}
	}
	limiter := cfg.RefreshUnknownKID
	if limiter == nil {
		limiter = rate.NewLimiter(rate.Every(time.Minute), 1)
	}
	noErrFirst := false
	keys, err := keyfunc.NewDefaultOverrideCtx(ctx, []string{cfg.JWKSURL}, keyfunc.Override{
		Client:            cfg.HTTPClient,
		RefreshUnknownKID: limiter,
		// 既定の 1 分だと、偽の kid を送り続けられたときに各要求が取り直しの順番を待って止まる。
		RateLimitWaitMax:          100 * time.Millisecond,
		NoErrorReturnFirstHTTPReq: &noErrFirst,
	})
	if err != nil {
		return nil, fmt.Errorf("libauth: jwks: %w", err)
	}
	return &Verifier{
		cfg:  cfg,
		keys: keys,
		parser: jwt.NewParser(
			jwt.WithValidMethods(algs),
			jwt.WithIssuer(cfg.Issuer),
			jwt.WithAudience(cfg.Audience),
			jwt.WithExpirationRequired(),
			jwt.WithIssuedAt(),
			jwt.WithLeeway(cfg.Leeway),
		),
	}, nil
}

type claims struct {
	jwt.RegisteredClaims
	ClientID string         `json:"client_id,omitempty"`
	Scope    string         `json:"scope,omitempty"`
	Scp      []string       `json:"scp,omitempty"`
	Cnf      map[string]any `json:"cnf,omitempty"`
}

// Verify はトークンを検証し、主体を返す。cert は TLS で提示されたクライアント証明書（無ければ nil）。
func (v *Verifier) Verify(ctx context.Context, token string, cert *x509.Certificate) (Principal, error) {
	var c claims
	_, err := v.parser.ParseWithClaims(token, &c, v.keys.KeyfuncCtx(ctx))
	if err != nil {
		return Principal{}, fmt.Errorf("%w: %w", ErrInvalidToken, err)
	}
	p := Principal{Subject: c.Subject, ClientID: c.ClientID, Scopes: scopes(c)}
	if cert != nil {
		p.CertThumbprint = Thumbprint(cert)
	}
	if v.cfg.RequireCertBinding {
		bound, _ := c.Cnf["x5t#S256"].(string)
		if cert == nil || bound == "" || subtle.ConstantTimeCompare([]byte(bound), []byte(p.CertThumbprint)) != 1 {
			return Principal{}, ErrCertBinding
		}
	}
	return p, nil
}

func scopes(c claims) []string {
	if len(c.Scp) > 0 {
		return c.Scp
	}
	return strings.Fields(c.Scope)
}
