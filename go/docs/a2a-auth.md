# A2A の 2 要素の認証

adk-agent と genkit-agent は、環境変数 `A2A_ADDR` を設定すると、Connect の口とは別に A2A の口を出す。
A2A の口は次の 2 つの要素を両方求める。

1. mTLS のクライアント証明書（TLS の段階で確かめる。無ければ接続を拒否する）
2. その証明書に結び付いた OAuth 2.0 のアクセストークン（RFC 8705。トークンの `cnf.x5t#S256` と提示した証明書の一致を確かめる）

盗んだトークンを別の証明書で使っても、証明書だけで呼んでも、どちらも 401 になる。
Agent Card（`/.well-known/agent-card.json`）だけは、トークンなしで取れる（証明書は要る）。

ツールの権限は、トークンのスコープで決める（`internal/a2aserve` の `Policy`）。
Policy に無いツール（MCP で足したものなど）は、A2A からは使えない。Session の State は権限の判断に使わない。

## 環境変数（受ける側）

| 変数 | 中身 |
| --- | --- |
| `A2A_ADDR` | 待ち受けるアドレス。未設定なら A2A の口を出さない |
| `A2A_AGENT` | 公開するエージェント（`research` / `operations`、既定 `operations`） |
| `A2A_BASE_URL` | Agent Card に書く URL（既定 `https://<A2A_ADDR>`） |
| `A2A_TLS_CERT` / `A2A_TLS_KEY` | サーバー証明書と鍵 |
| `A2A_CLIENT_CA` | クライアント証明書を確かめる CA |
| `A2A_JWKS_URL` / `A2A_ISSUER` / `A2A_AUDIENCE` | トークンの検証 |
| `A2A_IDP_CA` | IdP を私設の CA で立てたときの CA |
| `A2A_TOKEN_URL` | Agent Card に書くトークンの口 |
| `A2A_REQUIRED_SCOPE` | A2A を呼ぶのに要るスコープ（既定 `a2a.invoke`） |
| `A2A_DEV_ALLOW_UNBOUND_TOKENS` | `true` で証明書に結び付かないトークンも通す。開発用。mTLS は外れない |

`A2A_ADDR` を設定して、ほかの必須の変数が欠けると起動を止める。

## 手元で動かす

```
# 1. 認証局、証明書、IdP（証明書と鍵は ../ops/.run/pki に書く。git には入らない）
go run ./cmd/dev-idp -dir ../ops/.run/pki -server genkit-agent -server adk-agent \
    -client adk-agent=a2a.invoke,orders.read -client genkit-agent=a2a.invoke,orders.read

# 2. genkit-agent の A2A の口
PKI=../ops/.run/pki A2A_ADDR=localhost:19940 \
A2A_TLS_CERT=$PKI/genkit-agent-server.pem A2A_TLS_KEY=$PKI/genkit-agent-server.key \
A2A_CLIENT_CA=$PKI/ca.pem A2A_IDP_CA=$PKI/ca.pem \
A2A_JWKS_URL=https://127.0.0.1:19800/.well-known/jwks.json A2A_ISSUER=https://127.0.0.1:19800 \
A2A_AUDIENCE=a2a://agents A2A_TOKEN_URL=https://127.0.0.1:19800/token \
go run ./cmd/genkit-agent

# 3. adk-agent のクライアント証明書で呼ぶ
go run ./cmd/a2a-call -url https://localhost:19940 -message "注文 A-1 を見せて" \
    -cert ../ops/.run/pki/adk-agent-client.pem -key ../ops/.run/pki/adk-agent-client.key \
    -ca ../ops/.run/pki/ca.pem -client-id adk-agent
```

ADK のエージェントから別のエージェントを呼ぶ例は `internal/a2aserve/a2aserve_test.go` の `TestADKAgentCallsGenkitAgentWithClientCredentials`（`remoteagent.NewA2A` に、`libauth.ClientCredentials` の http.Client を渡す）。

## ほかのリポジトリへ移すとき

`internal/lib/libauth`（検証、ミドルウェア、TLS、Client Credentials）はフレームワークに依存しない。
`internal/toolscope`（ADK のプラグインと Genkit のミドルウェア）と `internal/a2aserve` の `Policy` / `Scopes` を、移す先のツールに合わせて書き換える。
`internal/devidp` と `cmd/dev-idp` は手元の確認用で、本番では実際の IdP（mTLS のクライアント認証と証明書に結び付いたトークンに対応するもの）を使う。
