# playground

書籍やブログ記事の内容を実際に動かして検証するスクラッチ領域。本格的なエージェントコードは `core/` / `cli/` / `agents/` に置く。`playground/` のファイルは「使い捨ての検証コード」として扱い、長期保守はしない。

## simple-call.ts

3 プロバイダー（OpenAI / Anthropic / Google）の LLM API を SDK を使わず **raw fetch** で呼ぶデモ。各社のリクエスト / レスポンス形式の差を直接見るため。

### 実行

```bash
# .env を用意（cp .env.example .env して使うキーを埋める）
bun run playground/simple-call.ts anthropic
bun run playground/simple-call.ts openai
bun run playground/simple-call.ts google
```

### 各社の差が見えるポイント

| 観点 | OpenAI | Anthropic | Google |
|---|---|---|---|
| 認証ヘッダ | `Authorization: Bearer ...` | `x-api-key` + `anthropic-version` | クエリパラメータ `?key=...` |
| エンドポイント | `/v1/chat/completions` | `/v1/messages` | `/v1beta/models/{name}:generateContent` |
| messages 構造 | `messages: [{role, content}]` | `messages: [{role, content}]` + 別 `system` | `contents: [{role, parts: [{text}]}]` |
| max_tokens | 任意（`max_tokens` / `max_completion_tokens`） | 必須（`max_tokens`） | 任意（`generationConfig.maxOutputTokens`） |
| レスポンス | `choices[0].message.content` | `content[].text`（type=text のみ） | `candidates[0].content.parts[].text` |
| 終了理由 | `finish_reason` | `stop_reason` | `finishReason`（camelCase） |
| usage | `prompt_tokens` / `completion_tokens` | `input_tokens` / `output_tokens` | `promptTokenCount` / `candidatesTokenCount` |

`core/providers/anthropic.ts` 等の SDK 経由実装と比較すると、SDK が **「HTTP / 認証 / リトライ / エラー型 / ストリーミング」を肩代わりして、アプリ側は『プロバイダー間の概念差の変換』に集中** できることが見える。

### モデル選択

書籍に倣ってデフォルトは軽量モデル。

| プロバイダー | 軽量（デフォルト）| 高性能 |
|---|---|---|
| Anthropic | `claude-haiku-4-5-20251001` | `claude-sonnet-4-6` / `claude-opus-4-7` |
| OpenAI | `gpt-5-mini` | `gpt-5` |
| Google | `gemini-2.5-flash` | `gemini-2.5-pro` |

ハンズオンや動作確認は軽量で十分。複雑なリファクタや複数ファイル横断の推論が必要なときだけ高性能モデルへ切り替える。
