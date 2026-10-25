# agent/python

Python での AI エージェント実装を、フレームワーク / アプローチ別にサブディレクトリに分けて並列管理するモノレポ。

## サブ実装

| ディレクトリ | アプローチ | 状態 |
|---|---|---|
| [`langchain/`](./langchain) | LangChain + LangGraph 流。`ChatPromptTemplate` / `MessagesPlaceholder` / `OutputParser` / `Document` 等のプリミティブをフル活用 | py-phase1 完了（doc_reader） |
| [`contrastive_negatives/`](./contrastive_negatives) | 対照学習（InfoNCE）の負例の扱い（バッチ内 / 一様 / logQ 補正 / 偽の負例の除外）を、合成した商品推薦の候補検索で比べる実験。numpy + torch | 実験と結果を README に掲載 |
| [`knowledge_extraction/`](./knowledge_extraction) | 架空の人物の伝記で小さな GPT を訓練し、伝記を覚えることと QA で取り出せることの差を、伝記の増強 / 線形プローブ / 逆転の呪いで測る実験。Physics of Language Models Part 3.1 の小さな再現。numpy + torch | 実験と結果を README に掲載 |

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
