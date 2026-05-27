# CLI 層

ローカル実行環境。LangGraph の `create_react_agent` を使って Agent を組み立てる + Tool 群を提供する。

## 責務

- Agent 構築（LangGraph `create_react_agent`）
  - `build_agent` — checkpointer なしの素の ReAct ループ
  - `build_hitl_agent` — `MemorySaver` を組み込み、Tool 内の `interrupt()` で人間承認を挟める版
- 標準 Tool 群（search_documents / web_search / write_file 等）
  - `write_file` は破壊的操作。実行前に `interrupt()` で停止し、ワークスペース配下に書き込みを閉じ込める

## 実装済（旧 py-phase3 候補）

- Memory（`MemorySaver` で checkpoint）— `build_hitl_agent`
- HITL（LangGraph の `interrupt()` + `Command(resume=...)`）— `cli.tools.write_file` + `agents.web_researcher`

## 未実装（候補）

- `SqliteSaver` での永続 checkpoint
- 独自 `StateGraph` への置き換え

## 依存

- Core 層（`core.providers`、`core.types`）
- LangGraph

## ファイル構成（予定）

```
cli/
├── __init__.py
├── agent.py              Agent factory（create_react_agent ラッパ）
├── tools/
│   ├── __init__.py
│   ├── read_file.py
│   ├── list_files.py
│   └── grep.py
└── README.md
```

## 設計方針

- まずは `create_react_agent` の prebuilt を使う（ts/ の自前ループと対比）
- 慣れたら `StateGraph` で独自構造を組む（py-phase4 以降）
- Tool は LangChain `BaseTool` で定義

## ts/ との対比

| 観点 | ts/（自前実装） | python/（LangGraph） |
|---|---|---|
| Agent ループ | `while step < maxSteps` を自前 | `create_react_agent` が裏でループ |
| 停止条件 | `finishReason==stop` + `maxSteps` + `tool_use` 不在 | LangGraph 内部で同等の判定 |
| State | `messages` 配列のみ | `StateGraph` の typed state |
| HITL | 自前で挟む（py-phase3 候補） | `interrupt()` で標準提供 |
| Memory | 自前（py-phase3 候補） | `MemorySaver` / `SqliteSaver` |

「ts/ で自前で書いたものが LangGraph では既に存在する」という対応関係を意識すると、フレームワークの恩恵と引き換えに失う透明性が見えてくる。
