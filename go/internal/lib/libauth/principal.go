// Package libauth はエージェント間の呼び出しを、mTLS のクライアント証明書と、その証明書に結び付いた
// OAuth 2.0 のアクセストークン（RFC 8705）の 2 要素で認証する。
//
// 受ける側は Verifier と Middleware、呼ぶ側は ClientCredentials、TLS の設定は ServerTLSConfig / ClientTLSConfig を使う。
// 認証した主体は context に載せ、ツールの権限の判断はその主体のスコープで行う。State は使わない。
package libauth

import (
	"context"
	"crypto/sha256"
	"crypto/x509"
	"encoding/base64"
	"slices"
)

// Principal は認証した呼び出し元。トークンそのものは持たない。
type Principal struct {
	Subject  string
	ClientID string
	Scopes   []string
	// CertThumbprint は提示されたクライアント証明書の x5t#S256。
	CertThumbprint string
}

// HasScopes は required をすべて持つかを返す。
func (p Principal) HasScopes(required ...string) bool {
	for _, s := range required {
		if !slices.Contains(p.Scopes, s) {
			return false
		}
	}
	return true
}

type principalKey struct{}

// WithPrincipal は主体を context に載せる。
func WithPrincipal(ctx context.Context, p Principal) context.Context {
	return context.WithValue(ctx, principalKey{}, p)
}

// FromContext は context の主体を返す。
func FromContext(ctx context.Context) (Principal, bool) {
	p, ok := ctx.Value(principalKey{}).(Principal)
	return p, ok
}

// Thumbprint は証明書の DER の SHA-256 を base64url（パディングなし）にした x5t#S256 を返す。
func Thumbprint(cert *x509.Certificate) string {
	sum := sha256.Sum256(cert.Raw)
	return base64.RawURLEncoding.EncodeToString(sum[:])
}
