# agent/ts — TypeScript + Bun のフレームワーク非依存エージェント実装

LangChain / Vercel AI SDK / Mastra 等のフレームワークを使わず、TypeScript + Bun で AI エージェントをゼロから組み立てる実験。`go/` の Genkit 実装と並列して、同じ抽象化を異なる言語でどう表現するかを比較する。

## 哲学

**人間が "what"、エージェントが "how"**

イシュー駆動の開発ワークフローでは、人間が「何を作りたいか」「何を読みたいか」を Issue として書き、エージェントが「どうやって調べるか」「どうやって書くか」を担う。定型的なタスクは自動化し、人間は創造的な意思決定に集中する。

過剰自動化の罠を避けるため:

- PR を作るところまでは自動化、merge は人間が握る
- 破壊的操作には HITL（Human-in-the-loop）承認を挟む
- エージェントは「いい感じ」で許される領域に閉じ、厳密性が必要なところはスクリプト or 人間に残す

## アーキテクチャ（3 層）

```
┌─────────────────────────────────────────────────┐
│ Action 層 — CI/CD 統合                           │
│   GitHub Actions エントリ                        │
│   Issue → コード修正 → PR 作成                   │
├─────────────────────────────────────────────────┤
│ CLI 層 — ローカル実行環境                        │
│   Agent クラス + 思考ループ                      │
│   Tool 群（readFile / listFiles / grep / 等）    │
│   サンドボックス（bwrap / Docker）               │
│   Context compaction                             │
├─────────────────────────────────────────────────┤
│ Core 層 — LLM API 抽象化                         │
│   LanguageModel SPI（doGenerate / doStream）     │
│   Provider 実装（Anthropic / Google / OpenAI）   │
│   共通型（Tool / Message / ToolCall / Error）    │
└─────────────────────────────────────────────────┘
```

依存関係は **Core ← CLI ← Action** の一方向のみ。Core は外部依存ゼロ、CLI は Core に依存、Action は CLI に依存する。

## ディレクトリ構造

```
ts/
├── core/             LLM API 抽象化
│   ├── types.ts
│   ├── providers/
│   └── README.md
│
├── cli/              ローカル実行環境
│   ├── agent.ts
│   ├── tools/
│   ├── sandbox.ts
│   ├── manageContext.ts
│   └── README.md
│
├── action/           CI/CD 統合
│   ├── github.ts
│   ├── issueResolver.ts
│   └── README.md
│
├── agents/           個別エージェント
│   ├── repo-reader/
│   └── README.md
│
└── bin/              CLI / Action エントリポイント
    ├── repo-reader.ts
    └── repo-reader-action.ts
```

## エージェント追加の作法

新しいエージェントを足す時は `agents/{name}/` にディレクトリを作る。Core / CLI / Action 層は触らない。

```
agents/{name}/
├── prompt.ts       system prompt
├── runner.ts       本体（CLI からも Action からも呼ばれる）
└── output.ts       結果の整形・書き出し
```

詳細は `agents/README.md` を参照。

## 計画中のエージェント

| エージェント | 用途 |
|---|---|
| repo-reader | OSS リポを読み解いて要約を生成 |
| dev-digest | URL / PDF を要約してナレッジ化 |
| memory-curator | `.claude/memory/` の重複検出・整理 |
| interview-coach | 面接対策の Q&A 生成 |

## セットアップ

```bash
bun install
```

エージェントによって異なるが、最低限以下のいずれかが必要。

```bash
export ANTHROPIC_API_KEY=...
# or
export OPENAI_API_KEY=...
# or
export GOOGLE_GENAI_API_KEY=...
```

## ステータス

開発中。最初のエージェントとして `repo-reader` を実装予定（ts-phase1）。

## 参考

- nano-code: https://github.com/laiso/nano-code
- Building Effective AI Agents (Anthropic): https://www.anthropic.com/research/building-effective-agents
