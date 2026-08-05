# adk-agent 実装計画 — AgentService の ADK 版（genkit 版との比較実装）

genkit-agent（`docs/genkit-agent.md`）と同じ proto（AgentService）を ADK で実装する計画。
言語は Go（`google.golang.org/adk/v2`）。2026-07 時点の事実確認に基づく。

## 言語の決定 — Go

| 判断材料 | 事実 |
|---|---|
| adk-go の成熟度 | 2.0.0 GA（2026-06-30）、v2.1.0 現在。MCP / text streaming / マルチエージェント / HITL confirmation / VertexAiSessionService まで実用段階 |
| 唯一の欠落 | Live API（音声双方向）のみ Go 未実装（adk-go issue #550） |
| 比較の質 | genkit 版と同一言語・同一 proto・同一バックエンドで並ぶため、差分がフレームワークの差だけになる |
| 既存資産 | `internal/adk`（usecase.Ask 比較、adk v0.6.0）で API の土地勘あり。v2 は import path が別（`/v2`）なので v0.6 と共存可能 |

Live API だけは Python が必要（Phase 4 参照）。

## genkit 版との対応表（設計の骨子）

| genkit-agent の要素 | ADK 版での実装 |
|---|---|
| Definition + Registry（自作） | `llmagent.New` × 2（research / operations）。マルチエージェントはフレームワークのネイティブ機能 |
| DefineStreamingFlow + Generate | `runner.Run(ctx, userID, sessionID, msg, cfg)` が `iter.Seq2[*session.Event, error]` を返す。partial event を AskResponse.answer_delta に写像 |
| 承認フロー（pending ストア自作） | **ToolContext confirmation API**（`ctx.RequestConfirmation(message, payload)` / `ctx.ToolConfirmation()`、Go v0.3.0+、Experimental）。フレームワーク標準機能に置き換わるのが最大の比較ポイント |
| MCP（genkit plugins/mcp） | `tool/mcptoolset` の MCPToolset。接続先は同じ mcp/weather_go（Streamable HTTP） |
| session（Firestore サブコレクション自作） | 公式は InMemory / VertexAI / Database（RDB）のみで **Firestore 実装なし** → session.Service を自前実装（genkit 版 firestore.go の資産を流用） |
| メトリクス（transport で計測） | Before/After callback で計測し ask_completed ログを揃える（2.0 は _run_async_impl オーバーライド廃止、callback が正規手段） |
| Connect transport | **そのまま再利用**。Handler の依存を `Ask(ctx, input) iter.Seq2[*AskChunk, *AskOutput]` インターフェースに切り出し、genkit / ADK を差し替え可能にする |

## フェーズ分け

### Phase 1 — 骨格と streaming（AgentService 互換の最小形）— 実装済み 2026-08-05（docs/adk-agent.md）

1. `internal/agentcore` を新設し、AskInput / AskOutput / Ask インターフェースと Connect Handler を genkit-agent から抽出（genkit 版はこれの実装になる）
2. `internal/adkagent` — llmagent × 2、FunctionTool（order / geo / knowledge。バックエンドは genkitagent の InMemory 実装を共有）
3. Runner + InMemorySessionService で Ask（server streaming）を通す
4. `cmd/adk-agent`（PORT 19912）。cmd/genkit-ask がそのまま動作確認クライアントになる（proto 共通の利点の実証）

### Phase 2 — 承認・MCP・永続化

1. 書き込みツールを `RequestConfirmation` ベースに実装。ExecuteConfirmedToolCall RPC から confirmation 応答（`adk_request_confirmation` の function_response）を Runner に注入して再開する経路を設計（ここが最難所。REST の再開手順を Connect にどう写像するか、実装時に v2.1 の API で確定）
2. MCPToolset で weather_go に接続（Go 版の Streamable HTTP パラメータ型名は pkg.go.dev で実装時に確認）
3. Firestore SessionService 自前実装（session.Service インターフェース準拠）
4. callback でメトリクスログを genkit 版と同形式に

### Phase 3 — 評価（3 スタック比較の土台）

1. evalset（`.evalset.json`）を作成 — 「注文照会 → get_order が呼ばれる」「変更依頼 → 承認待ちになる」等のツール軌跡ケース
2. `adk eval` + `tool_trajectory_avg_score` / `response_match_score` を CI に組み込み
3. 同じケースを genkit 版にも流せる形（proto 経由の外形評価）に整え、フレームワーク非依存の比較レポートを書く

### Phase 4 — Live API（別トラック、任意）

- 音声双方向（run_live / LiveRequestQueue / RunConfig の BIDI）は Go 未実装のため、やるなら `agent/python/adk_live/` で最小デモ（テキスト server-streaming とは別系統であることをドキュメントに明記）
- Go 対応（issue #550）が動いたら Go に寄せる

## 事前予想（比較レポートで答え合わせする）

1. 承認フローは ADK が楽（confirmation API 標準）だが、Experimental ゆえの API 変更リスクを背負う
2. マルチエージェントは ADK が本業（workflow runtime / sub-agents）。genkit の Registry 自作より宣言的になる
3. セッション永続化は genkit の方が自由（ADK は session.Service の型に縛られる）
4. 評価は ADK の独走（adk eval 標準装備。genkit に相当物なし）

## リスクと前提

- confirmation API は Experimental。破壊的変更があり得る前提で transport 層に漏らさない
- ADK 2.0 は workflow runtime 移行で破壊的変更が多い。既存 `internal/adk`（v0.6.0）は触らず、/v2 併存で進める
- セッション DB スキーマは変更履歴あり（Python v1.22.0 で変更）。自前 Firestore 実装は Event スキーマ（node_info / output 追加）に追従する

## 出典

- https://adk.dev/2.0/ / https://pkg.go.dev/google.golang.org/adk/v2
- https://adk.dev/runtime/runconfig/（StreamingMode: NONE / SSE / BIDI）
- https://adk.dev/tools-custom/confirmation/（Tool Confirmation API）
- https://adk.dev/tools-custom/mcp-tools/（MCPToolset）
- https://adk.dev/sessions/session/（SessionService 3 実装、Firestore なし）
- https://adk.dev/evaluate/（adk eval / evalset）
- https://github.com/google/adk-go/issues/550（Live API Go 未実装)
