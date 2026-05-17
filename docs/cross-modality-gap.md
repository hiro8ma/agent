# Cross-modality Continuity — このリポと隣接 mcp の実装ギャップ

## TL;DR

agent/ts と隣接する mcp サーバー群を **「複数モダリティ・複数クライアントから同じ session を参照する仕組み」** の観点で評価。**重要発見: mcp/memory が既に堅牢実装あり** (SQLite + FTS5 + ONNX embeddings + 時間減衰 14 日半減期)。**13h で agent/ts と統合して「CLI で始めて Claude Desktop で続ける」MVP が完成**する。

## 教科書整理

### モダリティ統合

- 各モダリティに固有の強みと制約
- 最も魅力的な体験は複数モダリティを組み合わせた単一のシームレスなジャーニー
- モダリティ間を行き来しても「状態」と「コンテキスト」を途切れさせない

### シナリオ

| シナリオ | 内容 |
|---|---|
| A. 移動と作業の連動 | 運転中の音声 → スマホ text → デスクの GUI |
| B. 要約と詳細の切替 | 音声で要約、詳細は後から別チャネルへ |

## 2026 の収束パターン

- **ChatGPT**: cloud memory で Web / Desktop / Atlas (Android) を device-level 同期
- **Claude Projects**: ワークスペース単位で context 隔離
- **Gemini**: Google アカウントで Gmail / Drive / Calendar から自動 context 取り込み
- **ベンダー間 sync は native 未対応**、MemoryPlugin 等の third-party ブリッジが台頭
- **OpenAI Apps SDK**: MCP server が `structuredContent` + `content` + inline HTML widget、**data tool と render tool を分離**
- **Vercel AI SDK 3.0**: React Server Components で UI を LLM から stream
- **Microsoft Adaptive Cards**: JSON を Teams / Outlook / Copilot で host 別 native UI に変換、**2026 に AI agent の構造化 response 標準**
- **Intercom Fin / Zendesk omnichannel**: 会話履歴 + customer data + AI summary を agent に自動引き継ぎ、**「context 再構築させない」が CSAT 15-30% / FCR 20-35% 改善**
- **Stytch Web Bot Auth**: benign agent を暗号署名検証、**人間 + agent を同じ CIAM で扱う方向に収束**
- **2026 の標準**: **MCP を identity / UI / context の共通 bus に据え、structured payload を host 側で native render しつつ CIAM が人間と agent を一元管理**

## このリポの実装評価

### 1. Session 永続化 (agent/ts) — ❌ ない

**根拠**:
- `cli/agent.ts:31-35` — 各 `run()` 呼び出しで `messages` を再初期化、永続化層なし
- `core/types.ts` — Message / Tool / ToolCall 型は定義されるも **session ID 概念ゼロ**
- `agents/repo-reader/runner.ts:17-33` — 単発実行で `messages[]` は毎回ローカルメモリ

**優先 Top 2 ギャップ**:
1. **Session Manager** (`cli/session.ts` 新規 + `agent.ts` 拡張 / **6h**) — user_id / session_id / messages persistence を JSON/SQLite で実装
2. **TTL + Lifecycle** (`cli/session.ts` + `core/types.ts` / **4h**) — session expiry (24h default)、accessed_at 更新機構

### 2. Memory 機構 (mcp) — ✅ ある（ただし Cross-client 非対応）

**根拠**:
- `mcp/memory/memory_server.py:1-400+` — **SQLite + FTS5 + ONNX embeddings + 時間減衰（14 日半減期）で実装済み** ✅
- `mcp/memory/README.md:1-106` — hybrid search（FTS5 + vector similarity + reranking）、scope (category) 管理、4 tools (remember/recall/forget/memory_stats) 実装済み
- **問題**: MCP サーバーは単一ファイル DB (`memory.db`) ローカル参照、**複数 MCP client（Claude Code / Cursor / Claude Desktop）から同一 session を参照する global session ID 仕組みがない**

**優先 Top 2 ギャップ**:
1. **Cross-client Memory Namespace** (`mcp/memory/memory_server.py:163-204` の DB schema 拡張 / **3h**) — `user_id` / `client_id` column 追加、multi-tenant isolation
2. **Session Scope in Recall** (`mcp/memory/memory_server.py:254-375` / **2h**) — recall に `session_id` filter 追加、「CLI で始めて Web で続ける」時に同じ記憶を参照可能に

### 3. Cross-client / Cross-modality 連携 — 🟡 部分的（非常に限定的）

**根拠**:
- `mcp/memory/memory_server.py` — FastMCP サーバーは Claude Desktop config (mcp/claude_desktop_config.json) で登録可能
- `mcp/agent/mcp_agent.py` / `mcp/client/mcp_llm_client.py` — Python MCP client が存在し、memory サーバーに接続可能 ✅
- **問題**:
  - **agent/ts CLI は MCP サーバーを呼び出す仕組みがない**（tool は `cli/tools/readFile.ts` など local-only）
  - session ID / user ID 共通化なし → agent/ts CLI → mcp memory → Claude Desktop という「連鎖」が成立しない
  - 履歴永続化が agent/ts には全くなく、session jump 不可

**優先 Top 2 ギャップ**:
1. **Agent CLI ← MCP Memory 連携** (`agent/ts/cli/mcp-client.ts` 新規 + `agent.ts` 統合 / **8h**) — agent/ts から `remember()` / `recall()` を MCP memory に delegate、conversation context 外部保存
2. **Shared Session ID / User ID** (`agent/ts/core/types.ts` + `mcp/memory/memory_server.py` / **5h**) — 全モダリティで `user_id` / `session_id` ヘッダ / 環境変数で統一

## 総合評価

| 観点 | 判定 | 根拠 (file:line) | Top 2 ギャップ（工数 h） |
|---|---|---|---|
| **Session 永続化** | ❌ | `agent/ts/cli/agent.ts:31-35` 毎回初期化 | session.ts (6h) / TTL lifecycle (4h) |
| **Memory 機構** | ✅ | `mcp/memory/memory_server.py:1-400` 実装済み | Cross-client namespace (3h) / Session scope (2h) |
| **Cross-client 連携** | 🟡 | `mcp/agent/mcp_agent.py` 存在も `agent/ts` 統合なし | MCP memory delegate (8h) / Shared ID (5h) |

## 結論と実装ロードマップ

**重要発見**: **mcp/memory サーバー（FastMCP, SQLite + FTS5）は堅牢だが Cross-client は構造化されていない**。user_id / session_id / client_id をスキーマに入れれば multi-tenant 対応可能。

### 実装優先順 (合計 28h)

1. **Phase 0 (10h)**: agent/ts に session 永続化追加（Session Manager 6h + TTL 4h）
2. **Phase 1 (5h)**: mcp/memory に user_id / session_id namespace 追加（Cross-client 3h + Session scope 2h）
3. **Phase 2 (13h)**: agent/ts ↔ mcp memory 統合（MCP memory delegate 8h + Shared ID 5h）
4. **Phase 3 (任意)**: portfolio の Web 検索ログを mcp memory に取り込み、Web ↔ CLI 連携

**28h で「CLI / Web / Claude Desktop を跨ぐ Cross-modality continuity」が個人プロジェクトで完成**。

## 累積 6 docs 総合

- four-tradeoffs-gap-analysis.md: 26h
- architecture-and-eval-gap.md: 112h
- llmops-gap-analysis.md: 9h
- hitl-rollout-kpi-gap.md: 65h
- ux-modality-gap.md: 22h
- gui-voice-video-gap.md: 44h
- **cross-modality-gap.md（本ファイル）: 28h**
- **累積合計: 306h**

Production-ready なエージェント実装の全体像が見える。**Cross-modality continuity は mcp/memory という既存資産を活かして「他観点より低コストで効果大」になる可能性が高い**。

## 参考

- [ChatGPT apps with sync (OpenAI)](https://help.openai.com/en/articles/10847137-chatgpt-apps-with-sync)
- [One Memory, Every AI Platform (MemoryPlugin)](https://www.memoryplugin.com/platforms)
- [Build your ChatGPT UI – Apps SDK](https://developers.openai.com/apps-sdk/build/chatgpt-ui)
- [Introducing AI SDK 3.0 (Vercel)](https://vercel.com/blog/ai-sdk-3-generative-ui)
- [Adaptive Cards Overview (Microsoft)](https://learn.microsoft.com/en-us/adaptive-cards/)
- [Zendesk omnichannel handoff (eesel AI)](https://www.eesel.ai/blog/zendesk-omnichannel-handoff)
- [Auth0 vs WorkOS CIAM 2026 (SSOJet)](https://ssojet.com/blog/auth0-vs-workos-ciam-2026)
