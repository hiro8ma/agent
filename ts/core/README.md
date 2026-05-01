# Core 層

LLM API 抽象化レイヤー。Provider（Anthropic / Google / OpenAI 等）の差異を吸収する SPI と、エージェント全体で共有する型定義を置く。

## 責務

- `LanguageModel` インタフェース（`doGenerate` / `doStream`）の定義
- 共通型（`Tool` / `Message` / `ToolCall` / `LLMApiError`）の定義
- 各 Provider 実装（Anthropic / Google / OpenAI）
- `modelFactory` で env 変数から Provider を選択

## 依存

- 外部依存ゼロ（Provider SDK は `providers/{name}.ts` 内に閉じ込める）
- 上位層（CLI / Action）から呼び出されるが、自身は何にも依存しない

## ファイル構成（予定）

```
core/
├── types.ts              LanguageModel SPI / Tool / Message
├── providers/
│   ├── anthropic.ts
│   ├── google.ts
│   ├── openai.ts
│   └── factory.ts
└── index.ts
```

## 設計方針

- Provider 切替は env 変数 1 つで済むように（`LLM_PROVIDER=anthropic`）
- 型は `Tool` / `Message` 等を Vercel AI SDK 風に揃え、移行コストを下げる
- ストリーミングは AsyncIterable で表現
- エラーは Provider 固有エラーを `LLMApiError` でラップ
