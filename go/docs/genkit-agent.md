# genkit-agent — 部署別エージェント基盤（Genkit 版)

Genkit（Go）で「役割別エージェント × データストア × MCP × 承認付き実行 × メトリクス」を持つエージェント基盤。
genkit / ADK / genai の 3 スタックで同じものを作り比べる計画の genkit 版（先行実装）。
到達点は Gemini Enterprise 型の構成（役割別エージェントが共有データストアと MCP を使い、ログ / メトリクスが可視化に流れる）で、3 実装とも proto（AgentService）を共通契約にする。

既存の `internal/genkit`（usecase.Ask の実装比較）とは別物で、こちらはサービスとして起動する形。

## 実装済みの範囲（Gemini Enterprise の構成要素との対応）

| 構成要素 | この実装 | 実運用での差し替え先 |
|---|---|---|
| 役割別エージェント | `Definition`（system prompt + ツールの組合せ）× Registry。research / operations の 2 体 | Definition を増やすだけ |
| データストア | `search_knowledge` ツール + `internal/genkitagent/knowledge`（インメモリのキーワード検索） | Vertex AI Search / RAG Engine / pgvector |
| MCP（外部システム接続） | `MCP_SERVER_URL` を設定すると Streamable HTTP で接続しツールを自動登録（genkit plugins/mcp） | 任意の MCP サーバー（../../mcp/ のサーバー群など） |
| 承認付き実行 | 書き込み系ツールは承認待ち登録のみ → `ExecuteConfirmedToolCall` で人間承認後に実行。取り出しは一度きりで二重実行防止 | Propose→Verify→Authorize→Execute の Authorize 部分 |
| ログ / メトリクス | `ask_completed` 構造化ログ（レイテンシ・トークン・ツール数） | Cloud Logging → BigQuery sink |
| Web 検索 | 未実装 | MCP 経由の Web 検索ツールで代替可 |
| Agent Skills（手順書の遅延ロード） | `SKILLS_DIR` を設定すると SKILL.md をスキャンし、メタデータだけ system prompt に注入。本文は `use_skill` 呼び出し時にロード（genkit v1.11 の `middleware.Skills`） | スキルディレクトリを増やすだけ |

## 構成

```
proto/agent/v1/           # AgentService（ListAgents / Ask(stream) / ExecuteConfirmedToolCall）
gen/                      # buf generate の生成物
cmd/genkit-agent/         # サーバー本体（env config、手書き DI、h2c）
cmd/genkit-ask/           # 動作確認 CLI（-list / ask / -exec）
internal/agentcore/       # フレームワーク非依存の核。入出力型 / Agent インターフェース / Connect ハンドラ
internal/genkitagent/
├── agent/                # genkit 実装。Definition / flow / ツール / 承認 Executor（agentcore.Agent を満たす）
├── backend/              # ツール接続先のインメモリ実装（実運用は gRPC クライアント）
├── knowledge/            # データストアのインメモリ実装
└── session/              # 履歴（メッセージ分割）+ 承認待ちストア（InMemory / Firestore）
```

ADK 版（`docs/adk-agent.md`、`internal/adkagent/`）も同じ agentcore を実装し、transport と型を共有する。

- 履歴は Firestore サブコレクション（`agent_sessions/{id}/messages`）で 1 メッセージ 1 ドキュメント。1MB 上限を回避
- 429 リトライはチャンク未送出時のみ（送出後の再試行は先頭から重複するため）
- ツールのエラーは `{"error": ...}` で返してモデルに続きを判断させる

## 動かし方

```bash
VERTEX_PROJECT_ID=<gcp-project> go run ./cmd/genkit-agent
# オプション: FIRESTORE_PROJECT_ID（履歴・承認待ちの永続化）/ MCP_SERVER_URL（MCP ツール取り込み）
#            SKILLS_DIR（Agent Skills のディレクトリ。例 cmd/genkit-agent/skills）

go run ./cmd/genkit-ask -list
go run ./cmd/genkit-ask -agent operations -session s1 "注文 ord-001 の支払い方法を教えて"
go run ./cmd/genkit-ask -agent operations -session s1 "注文 ord-001 の支払い方法をクレジットカードに変更して"
# => [pending] が返る
go run ./cmd/genkit-ask -exec <toolCallId>   # 人間の承認に相当
go run ./cmd/genkit-ask -agent research -session s2 "リモートワークは週何日までできますか"
```

## 検証済み（2026-07-30、Vertex AI 実呼び出し）

- 照会: get_order 選択 → 「銀行振込です」
- 承認フロー: 変更依頼 → pending 登録 → exec で実行 → 再照会で「クレジットカード」に反映
- 二重実行: 同じ toolCallId の再実行は NotFound
- research: search_knowledge でナレッジから回答
- メトリクス: `ask_completed` ログにレイテンシ / トークン / ツール数

## proto 再生成

```bash
buf generate   # protoc-gen-go / protoc-gen-connect-go が PATH に必要
```

## MCP 実接続（2026-07-30 検証済み）

mcp リポの weather_go（公式 Go SDK 製、Open-Meteo）を Streamable HTTP で立て、research エージェントに接続して確認済み。

```bash
# 別ターミナルで MCP サーバーを起動
(cd ../../mcp/weather_go && go run . -http :19920)

MCP_SERVER_URL=http://localhost:19920 VERTEX_PROJECT_ID=<gcp-project> go run ./cmd/genkit-agent
go run ./cmd/genkit-ask -agent research "いまの東京の天気と気温を教えて"
# => internal-systems_get_current_weather が選択され、Open-Meteo の実データで回答
```

## Agent Skills 実接続（2026-08-04 検証済み）

genkit v1.11.0 の `middleware.Skills` で、SKILL.md 形式の手順書を遅延ロードする Agent Skills を有効化できる。
サンプルは `cmd/genkit-agent/skills/refund-escalation/`（返金・キャンセルの判断分岐）。

```bash
SKILLS_DIR=cmd/genkit-agent/skills VERTEX_PROJECT_ID=<gcp-project> go run ./cmd/genkit-agent
go run ./cmd/genkit-ask -agent operations "注文 ord-001 をキャンセルして返金してほしい。流れを教えて"
# => use_skill(refund-escalation) が呼ばれ、手順書の「5 営業日以内」「8 日以内」を使って回答
```

検証結果（Vertex AI 実呼び出し）

- 返金の依頼では `use_skill` が発火し、SKILL.md の判断基準（発送前 / 発送後 / 3 万円超のエスカレーション）に沿って回答した
- 無関係な照会（支払い方法の確認）では `use_skill` は呼ばれず `get_order` のみ。入力トークンはスキル本文ロード時 602 に対し未ロード時 323 で、メタデータ注入だけのコストに留まる
- system prompt に手順を書く方式と違い、手順書は Markdown ファイルとして版管理でき、ロードは必要時だけになる

`use_skill` はツールターンを 1 回消費するため、多段のツール利用と重なる場合は `askMaxTurns` の残りに注意。

## 次の実装計画

1. ADK 版・genai 版を同じ proto（AgentService）で実装し、3 スタック比較を書く
2. eval ハーネス（ツール選択の正答率・承認フローの回帰）

## 残論点

- 認可（誰がどのエージェント・どのツールを使えるか）は未実装。ツール allowlist とセットで設計する
- 承認待ちに TTL がない。実運用では期限切れ削除が必要
