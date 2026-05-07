# agent/python — LangChain / LangGraph 流の reference 実装

LangChain + LangGraph + MCP を組み合わせた学習用エージェント実装。`go/`（Genkit）、`ts/`（フレームワーク非依存）と並列して、3 言語 × 3 アプローチで「同じ抽象化を異なるエコシステムでどう表現するか」を比較する。

## 哲学

**人間が "what"、エージェントが "how"**

イシュー駆動の開発ワークフローでは、人間が「何を作りたいか」「何を読みたいか」を Issue として書き、エージェントが「どうやって調べるか」「どうやって書くか」を担う。定型的なタスクは自動化し、人間は創造的な意思決定に集中する。

過剰自動化の罠を避けるため:

- PR を作るところまでは自動化、merge は人間が握る
- 破壊的操作には HITL（Human-in-the-loop）承認を挟む
- エージェントは「いい感じ」で許される領域に閉じる

## 位置付け

このリポ内の他実装との比較:

| 実装 | スタンス |
|---|---|
| `go/` | Google エコシステム（Genkit / genai / ADK）の比較 |
| `ts/` | フレームワーク非依存（nano-code 流）、4 層を全部自分で組む |
| `python/`（本ディレクトリ） | LangChain / LangGraph エコシステムを正面から使う、フレームワークが裏で何をやっているか理解する |

LangChain は批判もある（重い抽象化、頻繁な breaking change）が、production 採用例 / コミュニティ規模では依然最大級。**動作原理を理解した上で使う / 使わないを判断する** ための学習基盤。

## アーキテクチャ（3 層）

```
┌─────────────────────────────────────────────────┐
│ Action 層 — CI/CD 統合                           │
│   GitHub Actions エントリ                        │
│   Issue → コード修正 → PR 作成                   │
├─────────────────────────────────────────────────┤
│ CLI 層 — ローカル実行環境                        │
│   LangGraph の create_react_agent                │
│   Tool 群（read_file / search_documents / 等）   │
│   Memory / HITL は py-phase3 候補                │
├─────────────────────────────────────────────────┤
│ Core 層 — LLM API 抽象化                         │
│   LangChain ChatModel SPI                        │
│   Provider（langchain-openai / google / anthropic）│
│   共通型（pydantic BaseModel）                   │
└─────────────────────────────────────────────────┘
```

依存関係は **Core ← CLI ← Action** の一方向。

## ディレクトリ構造

```
python/
├── core/             LLM API 抽象化
│   ├── __init__.py
│   ├── types.py
│   ├── providers/
│   └── README.md
│
├── cli/              ローカル実行環境
│   ├── __init__.py
│   ├── agent.py
│   ├── tools/
│   └── README.md
│
├── action/           CI/CD 統合
│   ├── __init__.py
│   ├── github.py
│   └── README.md
│
├── agents/           個別エージェント
│   ├── __init__.py
│   ├── repo_reader/
│   └── README.md
│
└── bin/              CLI / Action エントリポイント
    ├── repo_reader.py
    └── repo_reader_action.py
```

## エージェント追加の作法

新しいエージェントを足す時は `agents/{name}/` にディレクトリを作る。Core / CLI / Action 層は触らない。詳細は `agents/README.md` を参照。

## 計画中のエージェント

| エージェント | 用途 |
|---|---|
| repo_reader | OSS リポを読み解いて要約を生成 |
| dev_digest | URL / PDF を要約してナレッジ化 |
| memory_curator | `.claude/memory/` の重複検出・整理 |

ts/ と同じエージェントを Python / LangChain で実装することで、エコシステム比較の素材になる。

## セットアップ

```bash
uv sync
```

エージェントによって異なるが、最低限以下のいずれかが必要。

```bash
export OPENAI_API_KEY=...
# or
export ANTHROPIC_API_KEY=...
# or
export GOOGLE_API_KEY=...
```

## ステータス

開発中。最初のエージェントとして `repo_reader` を実装予定（py-phase1）。

## 参考

- LangGraph: https://github.com/langchain-ai/langgraph
- LangChain: https://github.com/langchain-ai/langchain
- Building Effective AI Agents (Anthropic): https://www.anthropic.com/research/building-effective-agents
