package libauth

import (
	"context"
	"crypto/tls"
	"net/http"
	"net/url"

	"golang.org/x/oauth2"
	"golang.org/x/oauth2/clientcredentials"
)

// ClientConfig は呼ぶ側の設定。
type ClientConfig struct {
	TokenURL string
	ClientID string
	// ClientSecret は mTLS でクライアントを認証する IdP（RFC 8705 の tls_client_auth）では空でよい。
	ClientSecret string
	Scopes       []string
	// EndpointParams は audience などを要求する IdP 向け。
	EndpointParams url.Values
	// TLS はトークンの取得と A2A の呼び出しの両方で使う。証明書に結び付いたトークンを使うなら、同じ証明書にする。
	TLS *tls.Config
}

// ClientCredentials は Client Credentials でトークンを取り、要求に Bearer で付ける http.Client を返す。
// トークンの取得と要求は同じ TLS の設定（同じクライアント証明書）で行う。
func ClientCredentials(ctx context.Context, cfg ClientConfig) *http.Client {
	base := &http.Client{Transport: &http.Transport{TLSClientConfig: cfg.TLS, ForceAttemptHTTP2: true}}
	cc := clientcredentials.Config{
		ClientID:       cfg.ClientID,
		ClientSecret:   cfg.ClientSecret,
		TokenURL:       cfg.TokenURL,
		Scopes:         cfg.Scopes,
		EndpointParams: cfg.EndpointParams,
		AuthStyle:      oauth2.AuthStyleInParams,
	}
	return cc.Client(context.WithValue(ctx, oauth2.HTTPClient, base))
}
