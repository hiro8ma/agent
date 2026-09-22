package libauth_test

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/golang-jwt/jwt/v5"

	"github.com/hiro8ma/agent/go/internal/devidp"
	"github.com/hiro8ma/agent/go/internal/lib/libauth"
)

const (
	issuer   = "https://idp.test"
	audience = "a2a://orders-agent"
	scope    = "a2a.invoke"
	cardPath = "/.well-known/agent-card.json"
)

type env struct {
	ca       *devidp.CA
	idp      *devidp.IdP
	idpURL   string
	verifier *libauth.Verifier
	clientA  *devidp.Issued
	clientB  *devidp.Issued
}

func setup(t *testing.T, binding bool) *env {
	t.Helper()
	ca, err := devidp.NewCA("test-ca")
	if err != nil {
		t.Fatal(err)
	}
	idp, err := devidp.New(issuer, audience, map[string][]string{
		"agent-a": {scope, "orders.read"},
		"agent-b": {"orders.read"},
	})
	if err != nil {
		t.Fatal(err)
	}
	srvCert, err := ca.IssueServer("idp")
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewUnstartedServer(idp.Handler())
	srv.TLS = &tls.Config{Certificates: []tls.Certificate{srvCert.TLS}, ClientCAs: ca.Pool(), ClientAuth: tls.VerifyClientCertIfGiven}
	srv.StartTLS()
	t.Cleanup(srv.Close)

	jwksClient := &http.Client{Transport: &http.Transport{TLSClientConfig: &tls.Config{RootCAs: ca.Pool()}}}
	v, err := libauth.NewVerifier(t.Context(), libauth.Config{
		JWKSURL: srv.URL + devidp.JWKSPath, Issuer: issuer, Audience: audience,
		RequireCertBinding: binding, HTTPClient: jwksClient,
	})
	if err != nil {
		t.Fatal(err)
	}
	a, err := ca.IssueClient("agent-a")
	if err != nil {
		t.Fatal(err)
	}
	b, err := ca.IssueClient("agent-b")
	if err != nil {
		t.Fatal(err)
	}
	return &env{ca: ca, idp: idp, idpURL: srv.URL, verifier: v, clientA: a, clientB: b}
}

func (e *env) claims(mod func(jwt.MapClaims)) jwt.MapClaims {
	c := e.idp.Claims("agent-a", libauth.Thumbprint(e.clientA.Cert), []string{scope})
	if mod != nil {
		mod(c)
	}
	return c
}

func (e *env) sign(t *testing.T, c jwt.MapClaims) string {
	t.Helper()
	tok, err := e.idp.Sign(c)
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

func foreignKey(t *testing.T) *ecdsa.PrivateKey {
	t.Helper()
	k, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	return k
}

func TestVerify(t *testing.T) {
	t.Parallel()
	e := setup(t, true)
	currentKID := func(t *testing.T) string {
		t.Helper()
		tok, _, err := jwt.NewParser().ParseUnverified(e.sign(t, e.claims(nil)), jwt.MapClaims{})
		if err != nil {
			t.Fatal(err)
		}
		return tok.Header["kid"].(string)
	}
	testCases := map[string]struct {
		token   func(t *testing.T) string
		cert    *x509.Certificate
		wantErr error
	}{
		"正しいトークンと結び付いた証明書で通る": {
			token: func(t *testing.T) string { return e.sign(t, e.claims(nil)) }, cert: e.clientA.Cert,
		},
		"期限切れは拒否": {
			token: func(t *testing.T) string {
				return e.sign(t, e.claims(func(c jwt.MapClaims) { c["exp"] = time.Now().Add(-time.Hour).Unix() }))
			},
			cert: e.clientA.Cert, wantErr: libauth.ErrInvalidToken,
		},
		"宛先の違うトークンは拒否": {
			token: func(t *testing.T) string {
				return e.sign(t, e.claims(func(c jwt.MapClaims) { c["aud"] = "a2a://other" }))
			},
			cert: e.clientA.Cert, wantErr: libauth.ErrInvalidToken,
		},
		"発行者の違うトークンは拒否": {
			token: func(t *testing.T) string {
				return e.sign(t, e.claims(func(c jwt.MapClaims) { c["iss"] = "https://evil.test" }))
			},
			cert: e.clientA.Cert, wantErr: libauth.ErrInvalidToken,
		},
		"alg=none は拒否": {
			token: func(t *testing.T) string {
				tok := jwt.NewWithClaims(jwt.SigningMethodNone, e.claims(nil))
				tok.Header["kid"] = currentKID(t)
				s, err := tok.SignedString(jwt.UnsafeAllowNoneSignatureType)
				if err != nil {
					t.Fatal(err)
				}
				return s
			},
			cert: e.clientA.Cert, wantErr: libauth.ErrInvalidToken,
		},
		"公開鍵を HS256 の秘密に使う取り違えは拒否": {
			token: func(t *testing.T) string {
				c := &http.Client{Transport: &http.Transport{TLSClientConfig: &tls.Config{RootCAs: e.ca.Pool()}}}
				req, err := http.NewRequestWithContext(t.Context(), http.MethodGet, e.idpURL+devidp.JWKSPath, nil)
				if err != nil {
					t.Fatal(err)
				}
				res, err := c.Do(req)
				if err != nil {
					t.Fatal(err)
				}
				jwks, err := io.ReadAll(res.Body)
				res.Body.Close()
				if err != nil {
					t.Fatal(err)
				}
				tok := jwt.NewWithClaims(jwt.SigningMethodHS256, e.claims(nil))
				tok.Header["kid"] = currentKID(t)
				s, err := tok.SignedString(jwks)
				if err != nil {
					t.Fatal(err)
				}
				return s
			},
			cert: e.clientA.Cert, wantErr: libauth.ErrInvalidToken,
		},
		"別の鍵で署名したトークンは拒否": {
			token: func(t *testing.T) string {
				tok := jwt.NewWithClaims(jwt.SigningMethodES256, e.claims(nil))
				tok.Header["kid"] = currentKID(t)
				s, err := tok.SignedString(foreignKey(t))
				if err != nil {
					t.Fatal(err)
				}
				return s
			},
			cert: e.clientA.Cert, wantErr: libauth.ErrInvalidToken,
		},
		"証明書が無ければ結び付けの検査で拒否": {
			token: func(t *testing.T) string { return e.sign(t, e.claims(nil)) }, wantErr: libauth.ErrCertBinding,
		},
		"盗んだトークンを別の証明書で使うと拒否": {
			token: func(t *testing.T) string { return e.sign(t, e.claims(nil)) }, cert: e.clientB.Cert, wantErr: libauth.ErrCertBinding,
		},
		"cnf の無いトークンは拒否": {
			token: func(t *testing.T) string {
				return e.sign(t, e.claims(func(c jwt.MapClaims) { delete(c, "cnf") }))
			},
			cert: e.clientA.Cert, wantErr: libauth.ErrCertBinding,
		},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			p, err := e.verifier.Verify(t.Context(), tc.token(t), tc.cert)
			if !errors.Is(err, tc.wantErr) {
				t.Fatalf("err = %v, want %v", err, tc.wantErr)
			}
			if tc.wantErr == nil && (p.Subject != "agent-a" || !p.HasScopes(scope)) {
				t.Errorf("principal = %+v", p)
			}
		})
	}
}

func TestUnboundTokenPassesOnlyWhenBindingIsDisabled(t *testing.T) {
	t.Parallel()
	e := setup(t, false)
	tok := e.sign(t, e.claims(func(c jwt.MapClaims) { delete(c, "cnf") }))
	if _, err := e.verifier.Verify(t.Context(), tok, nil); err != nil {
		t.Errorf("結び付けを求めない設定で拒否された: %v", err)
	}
}

func TestJWKSIsCachedAndRefreshedOnlyForUnknownKID(t *testing.T) {
	t.Parallel()
	e := setup(t, true)
	base := e.idp.JWKSRequests()
	for range 20 {
		if _, err := e.verifier.Verify(t.Context(), e.sign(t, e.claims(nil)), e.clientA.Cert); err != nil {
			t.Fatal(err)
		}
	}
	if got := e.idp.JWKSRequests(); got != base {
		t.Fatalf("同じ kid で JWKS を %d 回取り直した", got-base)
	}
	if err := e.idp.Rotate(); err != nil {
		t.Fatal(err)
	}
	if _, err := e.verifier.Verify(t.Context(), e.sign(t, e.claims(nil)), e.clientA.Cert); err != nil {
		t.Fatalf("鍵の入れ替えの後の新しい kid が通らない: %v", err)
	}
	afterRotate := e.idp.JWKSRequests()
	if afterRotate != base+1 {
		t.Errorf("知らない kid で JWKS を %d 回取った, want 1", afterRotate-base)
	}
	started := time.Now()
	for range 10 {
		tok := jwt.NewWithClaims(jwt.SigningMethodES256, e.claims(nil))
		tok.Header["kid"] = "forged-" + rand.Text()
		s, err := tok.SignedString(foreignKey(t))
		if err != nil {
			t.Fatal(err)
		}
		if _, err := e.verifier.Verify(t.Context(), s, e.clientA.Cert); !errors.Is(err, libauth.ErrInvalidToken) {
			t.Fatalf("偽の kid が通った: %v", err)
		}
	}
	if got := e.idp.JWKSRequests(); got != afterRotate {
		t.Errorf("偽の kid の連発で JWKS を %d 回取り直した（頻度の上限が効いていない）", got-afterRotate)
	}
	if elapsed := time.Since(started); elapsed > 3*time.Second {
		t.Errorf("偽の kid の検証に %v かかった（取り直しの順番を待って止まっている）", elapsed)
	}
}

func protected(t *testing.T, e *env, required ...string) *httptest.Server {
	t.Helper()
	cert, err := e.ca.IssueServer("orders-agent")
	if err != nil {
		t.Fatal(err)
	}
	h := libauth.Middleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		p, _ := libauth.FromContext(r.Context())
		_ = json.NewEncoder(w).Encode(p)
	}), e.verifier, []string{cardPath}, required...)
	srv := httptest.NewUnstartedServer(h)
	srv.TLS = &tls.Config{Certificates: []tls.Certificate{cert.TLS}, ClientCAs: e.ca.Pool(), ClientAuth: tls.RequireAndVerifyClientCert, MinVersion: tls.VersionTLS13}
	srv.StartTLS()
	t.Cleanup(srv.Close)
	return srv
}

func TestMiddlewareWithTwoFactors(t *testing.T) {
	t.Parallel()
	e := setup(t, true)
	srv := protected(t, e, scope)
	tlsFor := func(c *devidp.Issued) *tls.Config {
		cfg := &tls.Config{RootCAs: e.ca.Pool(), MinVersion: tls.VersionTLS13}
		if c != nil {
			cfg.Certificates = []tls.Certificate{c.TLS}
		}
		return cfg
	}
	plain := func(c *devidp.Issued) *http.Client {
		return &http.Client{Transport: &http.Transport{TLSClientConfig: tlsFor(c)}}
	}
	stolen, err := func() (string, error) {
		return e.idp.Sign(e.claims(nil))
	}()
	if err != nil {
		t.Fatal(err)
	}
	testCases := map[string]struct {
		client   func(t *testing.T) *http.Client
		path     string
		header   string
		wantCode int
		wantTLS  bool
	}{
		"正しい組（証明書と、それに結び付いたトークン）で通る": {
			client: func(t *testing.T) *http.Client {
				return libauth.ClientCredentials(t.Context(), libauth.ClientConfig{TokenURL: e.idpURL + devidp.TokenPath, ClientID: "agent-a", Scopes: []string{scope}, TLS: tlsFor(e.clientA)})
			},
			path: "/", wantCode: http.StatusOK,
		},
		"証明書だけ（トークン無し）は 401": {
			client: func(*testing.T) *http.Client { return plain(e.clientA) }, path: "/", wantCode: http.StatusUnauthorized,
		},
		"盗んだトークンを自分の証明書で使うと 401": {
			client: func(*testing.T) *http.Client { return plain(e.clientB) }, path: "/", header: "Bearer " + stolen, wantCode: http.StatusUnauthorized,
		},
		"スコープの足りないトークンは 403": {
			client: func(t *testing.T) *http.Client {
				return libauth.ClientCredentials(t.Context(), libauth.ClientConfig{TokenURL: e.idpURL + devidp.TokenPath, ClientID: "agent-b", TLS: tlsFor(e.clientB)})
			},
			path: "/", wantCode: http.StatusForbidden,
		},
		"Agent Card は認証なしで 200": {
			client: func(*testing.T) *http.Client { return plain(e.clientA) }, path: cardPath, wantCode: http.StatusOK,
		},
		"トークンだけ（証明書無し）は TLS で拒否": {
			client: func(*testing.T) *http.Client { return plain(nil) }, path: "/", header: "Bearer " + stolen, wantTLS: true,
		},
	}
	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			t.Parallel()
			req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, srv.URL+tc.path, strings.NewReader("{}"))
			if err != nil {
				t.Fatal(err)
			}
			if tc.header != "" {
				req.Header.Set("Authorization", tc.header)
			}
			res, err := tc.client(t).Do(req)
			if tc.wantTLS {
				if err == nil {
					res.Body.Close()
					t.Fatalf("証明書の無い接続が通った: %d", res.StatusCode)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			defer res.Body.Close()
			body, _ := io.ReadAll(res.Body)
			if res.StatusCode != tc.wantCode {
				t.Fatalf("status = %d, want %d: %s", res.StatusCode, tc.wantCode, body)
			}
			if tc.wantCode == http.StatusOK && tc.path == "/" && !strings.Contains(string(body), `"Subject":"agent-a"`) {
				t.Errorf("主体が context に載っていない: %s", body)
			}
			if strings.Contains(string(body), "eyJ") {
				t.Errorf("応答にトークンが出ている: %s", body)
			}
		})
	}
}

func TestIdPRequiresClientCertificate(t *testing.T) {
	t.Parallel()
	e := setup(t, true)
	c := &http.Client{Transport: &http.Transport{TLSClientConfig: &tls.Config{RootCAs: e.ca.Pool()}}}
	req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, e.idpURL+devidp.TokenPath, strings.NewReader("grant_type=client_credentials"))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	res, err := c.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()
	if res.StatusCode != http.StatusUnauthorized {
		t.Errorf("証明書の無いトークンの要求 = %d, want 401", res.StatusCode)
	}
}
