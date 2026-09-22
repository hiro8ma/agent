package devidp

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/golang-jwt/jwt/v5"

	"github.com/hiro8ma/agent/go/internal/lib/libauth"
)

// JWKSPath は公開鍵の置き場所。TokenPath はトークンの発行口。
const (
	JWKSPath  = "/.well-known/jwks.json"
	TokenPath = "/token"
)

// IdP はクライアント証明書の CN ごとに許可したスコープで、証明書に結び付いたトークンを発行する。
type IdP struct {
	issuer   string
	audience string
	clients  map[string][]string
	ttl      time.Duration

	mu   sync.RWMutex
	keys map[string]*ecdsa.PrivateKey
	kid  string

	jwksHits atomic.Int64
}

// New は IdP を作る。clients はクライアント証明書の CN と、許可するスコープ。
func New(issuer, audience string, clients map[string][]string) (*IdP, error) {
	i := &IdP{issuer: issuer, audience: audience, clients: clients, ttl: 5 * time.Minute, keys: map[string]*ecdsa.PrivateKey{}}
	return i, i.Rotate()
}

// Rotate は新しい署名の鍵を作り、以後の発行に使う。古い鍵も JWKS に残す。
func (i *IdP) Rotate() error {
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return err
	}
	kid := rand.Text()
	i.mu.Lock()
	defer i.mu.Unlock()
	i.keys[kid] = key
	i.kid = kid
	return nil
}

// JWKSRequests は JWKS を配った回数。
func (i *IdP) JWKSRequests() int64 { return i.jwksHits.Load() }

// Sign は現在の鍵で任意のクレームに署名する。テストで不正なトークンを作るのに使う。
func (i *IdP) Sign(claims jwt.MapClaims) (string, error) {
	i.mu.RLock()
	key, kid := i.keys[i.kid], i.kid
	i.mu.RUnlock()
	t := jwt.NewWithClaims(jwt.SigningMethodES256, claims)
	t.Header["kid"] = kid
	return t.SignedString(key)
}

// Claims は発行するトークンの標準のクレーム。
func (i *IdP) Claims(clientID, thumbprint string, scopes []string) jwt.MapClaims {
	now := time.Now()
	return jwt.MapClaims{
		"iss": i.issuer, "aud": i.audience, "sub": clientID, "client_id": clientID,
		"scope": strings.Join(scopes, " "), "iat": now.Unix(), "exp": now.Add(i.ttl).Unix(),
		"cnf": map[string]any{"x5t#S256": thumbprint},
	}
}

// Handler は JWKS とトークンの発行口を返す。トークンの発行はクライアント証明書を必須にする。
func (i *IdP) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET "+JWKSPath, i.serveJWKS)
	mux.HandleFunc("POST "+TokenPath, i.serveToken)
	return mux
}

func (i *IdP) serveJWKS(w http.ResponseWriter, _ *http.Request) {
	i.jwksHits.Add(1)
	i.mu.RLock()
	keys := make([]map[string]string, 0, len(i.keys))
	for kid, k := range i.keys {
		pub, err := k.PublicKey.ECDH()
		if err != nil {
			continue
		}
		point := pub.Bytes()
		keys = append(keys, map[string]string{
			"kty": "EC", "crv": "P-256", "alg": "ES256", "use": "sig", "kid": kid,
			"x": base64.RawURLEncoding.EncodeToString(point[1:33]),
			"y": base64.RawURLEncoding.EncodeToString(point[33:]),
		})
	}
	i.mu.RUnlock()
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"keys": keys})
}

func (i *IdP) serveToken(w http.ResponseWriter, r *http.Request) {
	if r.TLS == nil || len(r.TLS.PeerCertificates) == 0 {
		oauthError(w, http.StatusUnauthorized, "invalid_client", "client certificate is required")
		return
	}
	cert := r.TLS.PeerCertificates[0]
	client := cert.Subject.CommonName
	allowed, ok := i.clients[client]
	if !ok {
		oauthError(w, http.StatusUnauthorized, "invalid_client", "unknown client")
		return
	}
	if err := r.ParseForm(); err != nil || r.PostForm.Get("grant_type") != "client_credentials" {
		oauthError(w, http.StatusBadRequest, "unsupported_grant_type", "client_credentials only")
		return
	}
	if id := r.PostForm.Get("client_id"); id != "" && id != client {
		oauthError(w, http.StatusUnauthorized, "invalid_client", "client_id does not match the certificate")
		return
	}
	scopes := allowed
	if req := strings.Fields(r.PostForm.Get("scope")); len(req) > 0 {
		for _, s := range req {
			if !slices.Contains(allowed, s) {
				oauthError(w, http.StatusBadRequest, "invalid_scope", s)
				return
			}
		}
		scopes = req
	}
	token, err := i.Sign(i.Claims(client, libauth.Thumbprint(cert), scopes))
	if err != nil {
		oauthError(w, http.StatusInternalServerError, "server_error", "sign")
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"access_token": token, "token_type": "Bearer", "expires_in": int(i.ttl.Seconds()),
		"scope": strings.Join(scopes, " "),
	})
}

func oauthError(w http.ResponseWriter, status int, code, desc string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]string{"error": code, "error_description": desc})
}
