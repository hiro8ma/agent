package a2ainterop

import (
	"context"
	"crypto/subtle"
	"errors"
	"net/http"
	"slices"
	"strings"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2asrv"
)

// Verifier はトークンを検証し、そのトークンに許されたスコープを返す。
type Verifier func(ctx context.Context, token string) (scopes []string, err error)

// ErrInvalidToken はトークンが無効なことを表す。
var ErrInvalidToken = errors.New("a2ainterop: invalid token")

// RequireBearer は Agent Card の取得を除くすべての要求で Bearer トークンを検証し、required のスコープをすべて求める。
//
// NewHandler が返すハンドラ全体を包む。v1.0 の口だけを包むと、v0.3 の互換の口から認証なしで呼べる。
func RequireBearer(next http.Handler, verify Verifier, required ...string) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == a2asrv.WellKnownAgentCardPath {
			next.ServeHTTP(w, r)
			return
		}
		token, ok := strings.CutPrefix(r.Header.Get("Authorization"), "Bearer ")
		if !ok || token == "" {
			w.Header().Set("WWW-Authenticate", "Bearer")
			http.Error(w, "missing bearer token", http.StatusUnauthorized)
			return
		}
		scopes, err := verify(r.Context(), token)
		if err != nil {
			w.Header().Set("WWW-Authenticate", `Bearer error="invalid_token"`)
			http.Error(w, "invalid token", http.StatusUnauthorized)
			return
		}
		for _, s := range required {
			if !slices.Contains(scopes, s) {
				w.Header().Set("WWW-Authenticate", `Bearer error="insufficient_scope"`)
				http.Error(w, "insufficient scope", http.StatusForbidden)
				return
			}
		}
		next.ServeHTTP(w, r)
	})
}

// StaticToken は決まった 1 つのトークンだけを受け付ける。確認用で、IdP の発行したトークンの検証には使わない。
func StaticToken(token string, scopes ...string) Verifier {
	return func(_ context.Context, got string) ([]string, error) {
		if subtle.ConstantTimeCompare([]byte(got), []byte(token)) != 1 {
			return nil, ErrInvalidToken
		}
		return scopes, nil
	}
}

// DeclareOAuth2 は Agent Card に OAuth 2 の client credentials の方式と、すべての呼び出しに求めるスコープを書く。
//
// 方式（securitySchemes）だけでは要件にならず、a2a-sdk の AuthInterceptor は認証情報を付けない。要件（securityRequirements）も書く。
func DeclareOAuth2(card *a2a.AgentCard, name a2a.SecuritySchemeName, tokenURL string, scopes map[string]string, required ...string) {
	if card.SecuritySchemes == nil {
		card.SecuritySchemes = a2a.NamedSecuritySchemes{}
	}
	card.SecuritySchemes[name] = a2a.OAuth2SecurityScheme{
		Flows: a2a.ClientCredentialsOAuthFlow{TokenURL: tokenURL, Scopes: scopes},
	}
	card.SecurityRequirements = append(card.SecurityRequirements, a2a.SecurityRequirements{name: required})
}

// DeclareCertBoundOAuth2 は OAuth 2 の client credentials と mutualTLS の両方を、1 つの要件（AND）として Agent Card に書く。
//
// 証明書に結び付いたトークン（RFC 8705）を求めるサーバー向け。要件を分けて並べると OR になり、片方だけで足りると読まれる。
func DeclareCertBoundOAuth2(card *a2a.AgentCard, oauthName, mtlsName a2a.SecuritySchemeName, tokenURL string, scopes map[string]string, required ...string) {
	if card.SecuritySchemes == nil {
		card.SecuritySchemes = a2a.NamedSecuritySchemes{}
	}
	card.SecuritySchemes[oauthName] = a2a.OAuth2SecurityScheme{
		Flows: a2a.ClientCredentialsOAuthFlow{TokenURL: tokenURL, Scopes: scopes},
	}
	card.SecuritySchemes[mtlsName] = a2a.MutualTLSSecurityScheme{Description: "クライアント証明書。アクセストークンはこの証明書に結び付く"}
	card.SecurityRequirements = append(card.SecurityRequirements, a2a.SecurityRequirements{oauthName: required, mtlsName: {}})
}
