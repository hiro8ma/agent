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
│   Tool 群（search_documents / web_search / 等）  │
│   HITL（interrupt + MemorySaver）実装済          │
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

## エージェント

| エージェント | 用途 | 状態 |
|---|---|---|
| search_agent | PDF を Chroma に索引して検索 Tool で回答 | 実装済 |
| web_researcher | Web を Tavily で調べ HTML レポートを書き出す（書き込みは HITL 承認） | 実装済 |
| mcp_client | 自作 [`mcp/`](../../../mcp/) サーバーに接続し tool を動的取得して実行 | 実装済 |
| repo_reader | OSS リポを読み解いて要約を生成 | 計画中 |
| dev_digest | URL / PDF を要約してナレッジ化 | アイデア段階 |
| memory_curator | `.claude/memory/` の重複検出・整理 | アイデア段階 |

### web_researcher（HITL）

破壊的操作に人間の承認を挟む哲学を体現するエージェント。`web_search`（Tavily）で調べ、結果を HTML レポートにまとめ、`write_file` でワークスペースに保存する。`write_file` は実行前に LangGraph の `interrupt()` で停止し、承認（`Command(resume="approve")`）で書き込み、拒否でスキップする。ファイル書き込みはワークスペース配下に閉じ込め、絶対パスと `..` での脱出を拒否する。

CLI（ターミナルで承認）

```bash
export OPENAI_API_KEY=...
export TAVILY_API_KEY=...
uv run python bin/web_researcher.py --topic "2026 年の AI エージェント動向"
```

Streamlit GUI（ボタンで APPROVE / DENY）

```bash
uv run streamlit run bin/web_researcher_app.py
```

ts/ と同じエージェントを Python / LangChain で実装することで、エコシステム比較の素材になる。

### mcp_client（agent/ ↔ mcp/ 連携）

自作 MCP サーバー群 [`mcp/`](../../../mcp/) に `MultiServerMCPClient`（`langchain-mcp-adapters`）で接続し、`get_tools()` で tool を動的取得して LangGraph の `create_react_agent` にバインドして実行する。この 2 リポを繋ぐ初の統合。

接続先は read 系で副作用がなく API キー不要の `calc` サーバー（FastMCP, stdio）。各サーバーは独立した uv プロジェクトなので、`uv run --directory <server_dir> python <entrypoint>` で子プロセス起動する。`mcp/` の場所は `agent/` の隣を自動検出し、`.env` の `MCP_REPO_PATH` で上書きできる。

tool 一覧の表示（LLM 不要。stdio 接続と `get_tools()` の動作確認に使える）

```bash
uv run python bin/mcp_client.py --list
```

tool を使って回答（`OPENAI_API_KEY` が必要）

```bash
export OPENAI_API_KEY=...
uv run python bin/mcp_client.py --question "半径 7 の円の面積を求めて"
```

| 接続先サーバー | 起動コマンド | tool |
|---|---|---|
| `calc` | `uv run --directory <mcp>/calc python calculator_server.py` | `add`, `subtract`, `multiply`, `divide`, `power`, `square_root`, `circle_area` |

mcp/ 側のコードには手を入れていない。接続するだけで tool を取り込める点が MCP の要点。サーバーを足すときは `agents/mcp_client/runner.py` の `MultiServerMCPClient` 設定に 1 エントリ追加する。

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

開発中。`search_agent`（RAG）と `web_researcher`（HITL Web リサーチ + Streamlit GUI）を実装済。HITL は LangGraph の `interrupt()` + `MemorySaver` で実現し、破壊的操作（ファイル書き込み）の前に人間の承認を挟む。

## 参考

- LangGraph: https://github.com/langchain-ai/langgraph
- LangChain: https://github.com/langchain-ai/langchain
- Building Effective AI Agents (Anthropic): https://www.anthropic.com/research/building-effective-agents
