# agent/python

Python での AI エージェント実装を、フレームワーク / アプローチ別にサブディレクトリに分けて並列管理するモノレポ。

## サブ実装

| ディレクトリ | アプローチ | 状態 |
|---|---|---|
| [`langchain/`](./langchain) | LangChain + LangGraph 流。`ChatPromptTemplate` / `MessagesPlaceholder` / `OutputParser` / `Document` 等のプリミティブをフル活用 | py-phase1 完了（doc_reader） |

## 今後追加候補

| ディレクトリ | アプローチ |
|---|---|
| `pydantic-ai/` | 型安全な構造化出力に特化、FastAPI 風 API |
| `autogen/` (Microsoft Agent Framework) | マルチエージェント会話モデル |
| `crewai/` | Role-based マルチエージェント |
| `vanilla/` | フレームワーク非依存（ts/ の Python 版） |

## 設計方針

各サブディレクトリは **独立した Python プロジェクト**。

- 自前の `pyproject.toml` を持つ
- 自前の `README.md` で哲学 / アーキテクチャを説明
- 独立した `uv sync` / 仮想環境

ルート `agent/python/` 自体には Python コードを置かない（このディレクトリは「複数アプローチを並列管理する箱」）。

## 比較の意図

同じユースケース（例: doc_reader / repo_reader）を異なるフレームワークで実装すると、各フレームワークの設計思想 / 抽象化のクセ / コード量 / 依存の重さが見える。`agent/go/`（Go エコシステム）、`agent/ts/`（TS フレームワーク非依存）と合わせて、3 言語 × 複数アプローチで AI エージェント実装の選択判断の引き出しを増やす。
