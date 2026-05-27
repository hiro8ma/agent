# agent/mastra — Mastra（TypeScript エージェントフレームワーク）の学習トラック

Mastra を正面から使う学習トラック。`go/`（Genkit）、`ts/`（フレームワーク非依存）、`python/langchain/`（LangChain / LangGraph）と並列して、「同じ抽象化を異なるエコシステムでどう表現するか」を比較する。

## 哲学

**人間が "what"、エージェントが "how"**

人間が「何を作りたいか」を決め、エージェントが「どうやるか」を担う。フレームワークを使う場合も、**裏で何をやっているかを理解した上で使う / 使わないを判断する** ことを目的にする。

## 位置付け

このリポ内の他実装との比較

| 実装 | スタンス |
|---|---|
| `go/` | Google エコシステム（Genkit / genai / ADK）|
| `ts/` | フレームワーク非依存（nano-code 流）、Core / CLI / Action を全部自分で組む |
| `python/langchain/` | LangChain / LangGraph エコシステムを正面から使う |
| `mastra/`（本ディレクトリ）| TypeScript エージェントフレームワーク。Agent / Memory / Workflow / Tool を Mastra の標準構造で組む |

`ts/` が「フレームワークなしで 4 層を自分で組む」のに対し、`mastra/` は「TS のフレームワークが Agent / Memory / Workflow をどう抽象化しているか」を見る対の関係。

## 設計方針

`agent/ts/` とは独立した自己完結プロジェクト。

- 自前の `package.json` / `tsconfig.json` を持つ
- パッケージマネージャは npm（`ts/` は Bun、混在を避けるため分離）
- `agent/ts/` の Core / CLI / Action 層は触らない

## ディレクトリ構造

```
mastra/
├── src/
│   └── mastra/
│       ├── index.ts          Mastra インスタンス（agents / logger を登録）
│       ├── providers.ts      provider セレクタ（LLM_PROVIDER で切替え）
│       └── agents/
│           └── assistant.ts  最小エージェント（instructions のみ）
├── package.json
├── tsconfig.json
├── .env.example
└── .gitignore
```

## provider の切替え

`LLM_PROVIDER` 環境変数で provider を選ぶ。既定は OpenAI。

```bash
LLM_PROVIDER=openai   # 既定
LLM_PROVIDER=google   # Gemini
LLM_MODEL=gpt-4o-mini # モデル ID 明示（未指定なら provider 既定）
```

Anthropic / Vertex に差し替える場合は `src/mastra/providers.ts` に provider を足す。

```ts
import { anthropic } from "@ai-sdk/anthropic";       // case "anthropic": return anthropic(modelId ?? "claude-sonnet-4-5");
import { vertex } from "@ai-sdk/google-vertex";       // case "vertex": return vertex(modelId ?? "gemini-2.5-flash");
```

## セットアップ

```bash
npm install
cp .env.example .env   # API キーを設定
```

最低限、選んだ provider のキーが必要。

```bash
export OPENAI_API_KEY=...
# or
export GOOGLE_GENERATIVE_AI_API_KEY=...
```

## 起動コマンド

```bash
npm run dev        # mastra dev（ローカル playground / API サーバー）
npm run build      # mastra build（.mastra/output に成果物を出力）
npm run typecheck  # tsc --noEmit
```

## ステータス

環境構築のみ完了。

- 完了 — Mastra プロジェクト scaffold、provider セレクタ（OpenAI / Gemini）、最小エージェント 1 本（instructions のみ）、型チェック / `mastra build` 通過
- 後続 — ツール（外部システム連携）、Memory（`@mastra/memory` + `@mastra/libsql`）、Workflow、Eval

## 参考

- Mastra: https://github.com/mastra-ai/mastra
- AI SDK: https://github.com/vercel/ai
