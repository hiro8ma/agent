# ジェネレーティブ UI / 音声 / マルチモーダル — このリポへの実装ギャップ

## TL;DR

このリポ (TS/Bun) を **ジェネレーティブ UI / 音声インターフェース / マルチモーダル入力** の 3 観点で評価。3 観点とも ❌ ない。**本回固有の実装ギャップは合計 44h**。優先は **構造化出力（マルチモーダル foundation）→ マルチモーダル入力（image/video 解析）→ 音声（後期 nice-to-have）** の順。

## 教科書整理

### GUI とジェネレーティブ UI

- 視覚的アフォーダンス（プログレスバー / 色分け / 警告）
- ワークフロー可視化（複数ステップ、条件分岐）
- AI 機能統合（Cursor / Windsurf / Copilot）
- **ジェネレーティブ UI**: クエリに応じて動的に UI 要素・データ可視化を生成

### 音声インターフェース

- 強み: ハンズフリー、インクルーシビティ
- 課題: 割り込み処理、速度の壁（150-180 語/分 < 読む 250-300 語/分）
- OpenAI Realtime API 等の登場で「滑らかさ」と「能力」が劇的に向上

### ビデオインターフェース

- 強み: マルチチャネル表現力、感情共感
- 課題: 処理能力 / 帯域幅 / **不気味の谷** / プライバシー

## 2026 の収束パターン

### GUI / ジェネレーティブ UI

- **Vercel AI SDK `streamUI`**: React Server Components を tool 経由でストリーム送出（**UI Message Streaming が主流化**）
- **Claude Artifacts / ChatGPT Canvas / Perplexity Spaces** が **「split-screen 協働」パラダイム**を共通言語化
- **2026 春**: Claude Remote Control / Perplexity Computer が GUI 操作を AI 側に渡す
- **ワークフロー**: n8n 2.0 / Lindy / Zapier Agents が schema-driven レンダリング

### 音声

- **OpenAI gpt-realtime GA**: WebRTC / WebSocket / SIP、$0.30/min、**cached input 80x 節約**
- **Cartesia Sonic Turbo TTFA 40ms** / Hume Octave 150ms 感情対応 / Sesame OSS
- **Orchestrator**: LiveKit と Pipecat が二強
- **barge-in**: VAD + TTS cancel + LLM turn rollback の rollback semantics

### ビデオ

- HeyGen Avatar IV / Synthesia / Tavus Phoenix-3 の三極
- **「あえてスタイライズ」+ b-roll が CS / 教育で好まれる**
- **片方向ビデオ**（ユーザー側 camera off）が CS / ヘルスケアでデフォルト

**2026 標準**: 「テキスト + 構造化 UI」「WebRTC + barge-in rollback」「スタイライズ片方向ビデオ」が三領域共通

## このリポ (TS/Bun) の実装評価

### 1. ジェネレーティブ UI / 構造化出力サポート — ❌ ない

**根拠**:
- `core/types.ts:10-17` — `Tool.parameters` は raw JSON Schema、zod 依存なし（package.json に zod あるが未使用）
- `core/generate.ts` — `generateText()` はテキスト string 返却のみ、構造化 response 型なし
- `cli/agent.ts:52-69` — テキスト + toolCall の union のみ、React component / table / form template なし
- Vercel AI SDK 相当の `streamObject()` / `generateObject()` 未実装

**優先 Top 2 ギャップ**:
1. **zod/JSON Schema 統合 + 構造化出力型** (`core/types.ts` + `core/generate.ts` / **5h**) — `Tool.parameters` を zod から自動生成、`GenerateTextResult` に `structured: Record<string, unknown>` フィールド追加、validation layer
2. **streaming generator + UI template** (`core/types.ts` に `doStream()` + `cli/streaming.ts` / **7h**) — `AsyncIterable<StreamChunk>` 実装、table/card/form レンダリング関数（CLI text 表現）、React component factory stub

### 2. 音声インターフェースへの拡張余地 — ❌ ない

**根拠**:
- `core/types.ts:33-36` — Message 型は text/toolCall のみ、audio なし
- `core/providers/*.ts` — audio parameter 引き受け機構なし、multimodal API 呼び出し未対応
- CLI (`bin/repo-reader.ts:10-23`) — テキスト引数パーサのみ、audio ファイルフラグなし
- WebSocket / WebRTC、VAD、barge-in 関連コード 0 行

**優先 Top 2 ギャップ**:
1. **Audio Message 型 + STT/TTS 抽象** (`core/types.ts` に audio content type + `cli/audio.ts` / **6h**) — `{ role, audio: Uint8Array, format }` Message variant、STT/TTS provider interface（`doTranscribe()` / `doSynthesize()`）
2. **WebSocket streaming + VAD integration** (`cli/audioStream.ts` / **12h**) — OpenAI Realtime API client、WebRTC offer/answer、Silero VAD wrapper、barge-in フロー（user audio 中断検知 → agent output 停止）

### 3. マルチモーダル入力（画像 / 動画）への対応 — ❌ ない

**根拠**:
- `core/types.ts:33-36` — Message は text のみ、vision content なし
- `cli/tools/readFile.ts:1-66` — text ファイルのみ、画像 / 動画パース未対応
- `bin/repo-reader.ts:10-24` — `--path` フラグのみ、`--image` / `--video` なし
- `core/providers/anthropic.ts:36-68` — image_source mapping なし、multimodal input 処理なし

**優先 Top 2 ギャップ**:
1. **Vision Message 型 + image/video Tool** (`core/types.ts` + `cli/tools/readImage.ts` / **5h**) — `{ role, vision: { url|base64, mediaType } }` Message、`readImage` tool（base64 encoding + size check）、`readVideo` stub（frame extraction）、MIME 検証
2. **Multimodal provider 実装** (`core/providers/anthropic.ts` 拡張 / **6h**) — vision block mapping（claude-opus-4-1 以降）、image token counting、video 入力（max frame 秒）対応、出力に image_source 返却対応

## 既存 doc との重複整理

- **architecture-and-eval**: テスト (25h) / feature flag (10h) 既出 → **構造化出力の eval 層は本回固有**
- **llmops-gap-analysis**: observability (2h) / prompt caching (1h) 既出 → **streaming multimodal token counting は固有**
- **ux-modality-gap**: tool 前通知 (4h) / slash command (5h) 既出 → **音声・画像フロー UX は固有**

## 総合判定表

| 観点 | 判定 | 根拠 | Top 2 ギャップ（推定工数） |
|---|---|---|---|
| **ジェネレーティブ UI** | ❌ ない | `core/types.ts:10-17` raw schema、`cli/agent.ts:52-69` テキストのみ、streaming 層なし | zod/JSON Schema 統合 (5h) / streaming + template (7h) |
| **音声インターフェース** | ❌ ない | `core/types.ts:33-36` text-only、provider audio 未対応、CLI audio flag なし | Audio Message + STT/TTS 抽象 (6h) / WebSocket + VAD (12h) |
| **マルチモーダル入力** | ❌ ない | Message vision なし、readFile text 限定、provider vision mapping なし | Vision Message + image/video Tool (5h) / multimodal provider (6h) |

## 本回固有の実装ギャップ（既存 5 docs 非重複）

- ジェネレーティブ UI: **12h**
- 音声インターフェース: **18h**
- マルチモーダル入力: **11h**
- **合計 44h**

既存 5 docs（four-tradeoffs 26h + architecture-and-eval 112h + llmops 9h + hitl-rollout-kpi 65h + ux-modality 22h = 234h）に追加すると、**累積 278h** で Production-ready なエージェント実装の全体像が見える。

## 実装優先順（本回 3 観点）

1. **構造化出力**（マルチモーダル対応 foundation、12h） — 他全部の基盤
2. **マルチモーダル入力**（image/video 解析、11h） — Vision LLM 時代の標準
3. **音声**（後期 nice-to-have、18h） — 用途が限定的

## 参考

- [Vercel AI SDK 3 Generative UI](https://vercel.com/blog/ai-sdk-3-generative-ui)
- [Altar.io (Human-AI Collaboration Patterns)](https://altar.io/next-gen-of-human-ai-collaboration/)
- [OpenAI gpt-realtime GA](https://openai.com/index/introducing-gpt-realtime/)
- [Cartesia Sonic](https://cartesia.ai/sonic)
- [LiveKit (Turn Detection for Voice Agents)](https://livekit.com/blog/turn-detection-voice-agents-vad-endpointing-model-based-detection)
- [Tavus (HeyGen Pricing & Alternatives)](https://www.tavus.io/post/heygen-pricing-breakdown-best-alternatives)
- [Hailuoai (Uncanny Valley in AI Video)](https://hailuoai.video/pages/blog/uncanny-valley-effect-ai-video-explained)
- [Dev.to (Top 8 AI Workflow Tools 2026)](https://dev.to/korix/top-8-ai-workflow-automation-tools-compared-2026-5807)
