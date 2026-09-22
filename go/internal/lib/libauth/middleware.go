package libauth

import (
	"crypto/x509"
	"errors"
	"net/http"
	"slices"
	"strings"
)

// Middleware は public に無いパスで Bearer トークンを検証し、主体を context に載せる。required のスコープが足りなければ 403。
//
// トークンは検証にだけ使い、ログにも context にも残さない。
func Middleware(next http.Handler, v *Verifier, public []string, required ...string) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if slices.Contains(public, r.URL.Path) {
			next.ServeHTTP(w, r)
			return
		}
		token, ok := strings.CutPrefix(r.Header.Get("Authorization"), "Bearer ")
		if !ok || token == "" {
			w.Header().Set("WWW-Authenticate", "Bearer")
			http.Error(w, "missing bearer token", http.StatusUnauthorized)
			return
		}
		var cert *x509.Certificate
		if r.TLS != nil && len(r.TLS.PeerCertificates) > 0 {
			cert = r.TLS.PeerCertificates[0]
		}
		p, err := v.Verify(r.Context(), token, cert)
		if err != nil {
			code := `Bearer error="invalid_token"`
			if errors.Is(err, ErrCertBinding) {
				code = `Bearer error="invalid_token", error_description="certificate binding mismatch"`
			}
			w.Header().Set("WWW-Authenticate", code)
			http.Error(w, "invalid token", http.StatusUnauthorized)
			return
		}
		if !p.HasScopes(required...) {
			w.Header().Set("WWW-Authenticate", `Bearer error="insufficient_scope"`)
			http.Error(w, "insufficient scope", http.StatusForbidden)
			return
		}
		next.ServeHTTP(w, r.WithContext(WithPrincipal(r.Context(), p)))
	})
}
