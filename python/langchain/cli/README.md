# CLI 層

ローカル実行環境。LangGraph の `create_react_agent` を使って Agent を組み立てる + Tool 群を提供する。

## 責務

- Agent 構築（LangGraph `create_react_agent`）
- 標準 Tool 群（read_file / list_files / grep / search_documents 等）

## py-phase3 候補（未実装）

- Memory（`MemorySaver` / `SqliteSaver` で checkpoint）
- HITL（LangGraph の `interrupt` API）
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
