# adk-agent — 部署別エージェント基盤（ADK 版）

genkit-agent（`docs/genkit-agent.md`）と同じ proto（AgentService）を ADK（`google.golang.org/adk/v2` v2.1.0）で実装したもの。
計画は `docs/adk-agent-plan.md`。Phase 1（骨格 + streaming）まで実装済み。

## 構成

```
internal/agentcore/       # フレームワーク非依存の核（Phase 1 で genkit-agent から抽出）
├── types.go              # AskInput / AskOutput / AskChunk などの入出力型
├── domain.go             # Order / KnowledgeDoc とサービスインターフェース
├── registry.go           # Agent インターフェース / Registry / SessionStore / ToolExecutor
├── connect.go            # Connect RPC ハンドラ（ask_completed メトリクスログ込み）
└── budget.go             # トークン予算（セッション単位 / プロセス全体の上限）
internal/adkagent/        # ADK 実装。agentcore.Agent を満たす
├── agent.go              # Definition / llmagent × Runner / イベント列 → チャンク写像
└── tools.go              # functiontool（get_order / resolve_area_names / search_knowledge）
cmd/adk-agent/            # サーバー本体（PORT 19912）
```

トークン予算（`agentcore/budget.go`）は Connect ハンドラ側にあるため、genkit 版と ADK 版で同じ env・同じ挙動になる。
上限に達したあとの `Ask` は `resource_exhausted` で拒否し、消費量は `ask_completed` ログの `budget_session_used` / `budget_total_used` に出る。

genkit 版・ADK 版とも `agentcore.Agent` インターフェース（`Ask(ctx, input) iter.Seq2[*AskChunk, *AskOutput]`）の実装になり、Connect transport・型・バックエンド（backend / knowledge のインメモリ実装）を共有する。
差分はフレームワークの使い方だけになり、比較が成立する。

## genkit 版との実装差分（Phase 1 時点）

| 要素 | genkit 版 | ADK 版 |
|---|---|---|
| エージェントループ | `genkit.Generate` + `WithMaxTurns` | `runner.Run` が内包（外からはイベント列） |
| streaming | `WithStreaming` コールバック | `RunConfig{StreamingMode: SSE}` の partial イベント |
| ツール定義 | `genkit.DefineTool`（ジェネリクス） | `functiontool.New`（ジェネリクス、ほぼ同形） |
| セッション履歴 | 自作ストア（InMemory / Firestore）から復元してプロンプトに詰める | ADK SessionService が管理（`AutoCreateSession: true`） |
| ツール記録 | 最終レスポンスの Request から抽出 | イベント列の FunctionCall パートから抽出。**partial と非 partial の両方に現れるため非 partial だけ記録する**（重複の罠） |

## 動かし方

```bash
VERTEX_PROJECT_ID=<gcp-project> go run ./cmd/adk-agent   # PORT 19912
GEMINI_API_KEY=<api-key> go run ./cmd/adk-agent          # GCP プロジェクト不要
# オプション: BUDGET_SESSION_TOKENS / BUDGET_TOTAL_TOKENS（トークン予算。未設定または 0 で無制限）

# クライアントは genkit-ask がそのまま使える（proto 共通の利点）
go run ./cmd/genkit-ask -url http://localhost:19912 -list
go run ./cmd/genkit-ask -url http://localhost:19912 -agent operations -session s1 "注文 ord-001 の支払い方法を教えて"
go run ./cmd/genkit-ask -url http://localhost:19912 -agent research "リモートワークは週何日までできますか"
```

## 検証済み（2026-08-05、Vertex AI 実呼び出し）

- ListAgents / Ask（server streaming）が genkit-ask からそのまま動作
- operations: get_order 選択 → 「銀行振込です」
- research: search_knowledge → 出典つきで回答
- セッション継続: 同一セッションで「その注文の金額は？」→ ツール再呼び出しなしで ADK 側の履歴から回答
- `ask_completed` ログ（レイテンシ / トークン / ツール数）が genkit 版と同形式で出力

## Phase 1 の残課題と次フェーズ

- 承認フロー: `functiontool.Config` の `RequireConfirmation` を確認済み。ExecuteConfirmedToolCall から Runner へ confirmation を注入する経路が Phase 2 の最難所
- MCP: `tool/mcptoolset` で weather_go に接続（Phase 2）
- セッション永続化: ADK に Firestore 実装がないため session.Service を自前実装（Phase 2）
- transport の履歴（agentcore.SessionStore）と ADK SessionService の履歴が二重管理になっている。Phase 2 で ADK 側に一本化する
