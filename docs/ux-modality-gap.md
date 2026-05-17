# エージェント UX (信頼性・発見可能性・記憶) — このリポへの実装ギャップ

## TL;DR

このリポを **信頼性・透明性 / 発見可能性とオンボーディング / コンテキスト・記憶の UX** の 3 観点で評価。信頼性 🟡 / 発見可能性 🟡 / 記憶 ❌。**本回固有の実装ギャップは合計 22h**。既存 docs（four-tradeoffs 26h + architecture 112h + llmops 9h + hitl 65h）に追加して、Production-ready なエージェント UX の積み重ねが見えるようになる。

## 教科書整理

### 信頼性と透明性（UX の基盤）

- 行動の予測可能性
- 理由の明確な説明

### インタラクションモダリティ

| モダリティ | 普及度 | 利用例 |
|---|---|---|
| テキスト | 非常に一般的 | カスタマーサポート、生産性 |
| GUI | 一般的 | ワークフロー調整、Cursor |
| 音声 | やや少ない | Alexa、コールセンター |
| ビデオ | まれ | バーチャル家庭教師 |

### テキストベース UX の課題

1. 発見可能性 (Discoverability) の低さ
2. 自然言語の曖昧さ

### テキスト UX 4 ベスプラ

1. オンボーディング（「以下が可能です」+ 具体例）
2. エラー管理 + コンテキスト保持
3. スムーズなターンテイキング
4. 簡潔・明確なレスポンス

## 2026 の収束パターン

### 信頼性・透明性

- **Cline**: telemetry なし、各 tool 呼び出しを平文で narrate
- **Claude Code**: SWE-bench **78.4%**、chain-of-thought と MCP tool trace を IDE 内に開示
- **Cursor**: inline diff で即時表示
- **Devin**: 非同期型、PR で結果説明
- **explainability**: 「思考 → tool → 差分」3 段表示が主流

### マルチモダリティ

- **音声**: Cartesia Sonic Turbo TTFA 40ms / Hume Octave 150ms / OpenAI gpt-realtime
- **GUI 側**: Cursor inline diff / Notion AI が suggestion chip で発見可能性を解決
- **切替 UX**: 「音声で指示 → GUI で差分提示」が主流

### コンテキスト管理

- **ChatGPT**: explicit + implicit
- **Claude Chat Memory**: 24h ごとに会話を synthesize、Settings で全件操作
- **Gemini Memory**: Google アカウント全体に接続
- **privacy 3 点セット**: 全件閲覧 + 個別削除 + temporary chat

### テキスト UX

- **OpenAI Apps SDK 原則**: 短く・list/table・即消化可能、**60 秒以内に実会話開始**
- 空 input の **ghost text**、**suggestion chip**
- **turn-taking**: 曖昧な時のみ 1 問返す原則
- **Smashing Magazine**: control / consent / accountability が agentic UX の 3 軸

**2026 標準**: 「思考 → tool → 差分」3 段可視化、Memory は scope + 削除権、modality 状況で切替、control・consent・accountability が共通言語

## このリポ (TS/Bun) の実装評価

### 1. 信頼性・透明性の UX — 🟡 部分的

**根拠**:
- 思考プロセス可視化: `cli/agent.ts:42-43` で `[agent] step N/maxSteps` のみ。「考えています」「Tool を呼び出しています」「結果を統合しています」の段階的説明なし
- Tool 呼び出し前表示: `cli/agent.ts:99` で実行後ログのみ。実行前の「これから tool_xxx を呼ぶ」告知なし
- エラー説明: `cli/agent.ts:96, 105` で単なる error string 返却。代替策提示なし
- Progress indicator: ステップ数のみ、streaming チャンク表示・% 進捗なし（`core/types.ts:68-70` に `doStream` 予定も未実装）

**優先 Top 2 ギャップ**:
1. **Tool 呼び出し前通知 + streaming メッセージ** (`cli/agent.ts` + `core/types.ts` / **4h**) — Tool 実行前に「Tool xxx(引数) を実行します」を表示、実行結果を逐次ストリーミング
2. **エラーハンドリング UI** (`cli/agent.ts:101-107` / **3h**) — 失敗時に「Tool xxx で失敗（理由: yyy）、代わりに zzz を試します」の提示

### 2. 発見可能性（Discoverability）と onboarding — 🟡 部分的

**根拠**:
- 起動時説明: `bin/repo-reader.ts:26-42` で `--help` のみ。起動メッセージ・デフォルト example なし
- slash command: なし（`--help` フラグのみ）
- Example prompts・suggestion chip: なし
- README.md: `ts/README.md:27-41` に使い方あり、具体例疎（「`bun run bin/repo-reader.ts --path .`」のみ）
- Tool 一覧表示: User が見れる仕組みなし

**優先 Top 2 ギャップ**:
1. **Tool 一覧 + example prompt** (`cli/agent.ts` + `bin/repo-reader.ts` / **2h**) — Agent 起動時に「利用可能 Tool: [readFile, listFiles]」+ 「example: 'Explore src/ and explain architecture'」を stdout 出力
2. **Slash command サポート** (`cli/agent.ts` に loop 層追加 / **5h**) — REPL 風に `/list-tools` `/clear` `/session` コマンド実装

### 3. コンテキスト・記憶の UX — ❌ ない

**根拠**:
- ターン数・履歴長表示: なし。`cli/agent.ts:31-91` で Message[] 管理するも表示機構なし
- 履歴クリア・セッション開始: なし。各 `agent.run()` 呼び出しで messages 再初期化（`cli/agent.ts:32-35`）、実質 REPL なし
- 履歴永続化: なし（JSON/SQLite/file）
- Memory に保存機能: なし

**優先 Top 2 ギャップ**:
1. **Session 管理・Conversation メモリ** (`cli/session.ts` 新規 + `agent.ts` 拡張 / **6h**) — `session.messages` に対話履歴を保持、`/clear` で消去、`session.turnCount()` を表示
2. **履歴永続化** (`cli/storage.ts` 新規 / **4h**) — session を JSON で `~/.agent-sessions/` に保存、load 時に復元

## 総合評価

| 観点 | 判定 | 根拠 | Top 2 ギャップ（推定工数） |
|---|---|---|---|
| 信頼性・透明性 | 🟡 部分的 | `cli/agent.ts:42-43` ステップ数のみ、`cli/agent.ts:96, 105` error string 返却 | Tool 前通知 + streaming (4h) / エラー UI (3h) |
| 発見可能性 | 🟡 部分的 | `bin/repo-reader.ts:26-42` `--help` のみ、slash command なし | Tool 一覧 + example (2h) / Slash command (5h) |
| 記憶 | ❌ ない | `cli/agent.ts:31-91` Message 管理あるが表示・永続化なし | Session 管理 (6h) / 履歴永続化 (4h) |

## 本回固有の実装ギャップ（既存 doc 非重複）

- 信頼性・透明性 UX: **7h**
- 発見可能性・onboarding: **7h**
- 記憶 UX: **10h**
- **合計 22h**

既存 4 docs（four-tradeoffs 26h + architecture-and-eval 112h + llmops 9h + hitl-rollout-kpi 65h）に追加すると、**Production-ready なエージェント UX の積み重ねが明確化される**。

## 参考

- [Codegen Blog (AI Coding Agents 2026)](https://codegen.com/blog/best-ai-coding-agents/)
- [Blink Blog (Best AI Coding Agents 2026)](https://blink.new/blog/best-ai-coding-agents-2026)
- [Cartesia Sonic](https://cartesia.ai/sonic)
- [Knightli (ChatGPT vs Claude Code vs Gemini Memory)](https://www.knightli.com/en/2026/05/07/chatgpt-claude-code-gemini-memory-comparison/)
- [OpenAI Apps SDK UX Principles](https://developers.openai.com/apps-sdk/concepts/ux-principles)
- [Smashing Magazine (Designing Agentic AI)](https://www.smashingmagazine.com/2026/02/designing-agentic-ai-practical-ux-patterns/)
- [GetStream (Chat UX)](https://getstream.io/blog/chat-ux/)
