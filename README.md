# AI Agent

AI エージェントの実装を複数言語・複数フレームワークで並べて比較するモノレポ

## Role in the Ecosystem

このリポは **Tool 消費層**。AI エージェント本体（LLM 呼び出し / Tool Dispatch ループ / 会話履歴 / プロンプト管理）を各言語・各フレームワークで実装する。エージェントが呼び出す Tool（MCP サーバー）は [`../mcp/`](../mcp/) を参照

## Implementations

| 言語 | ディレクトリ | 状態 | スタンス |
|---|---|---|---|
| Go | [`go/`](./go) | 実装済み | Google エコシステム（genai / Genkit / ADK）の比較 |
| TypeScript | [`ts/`](./ts) | ts-phase1 完了 | フレームワーク非依存（nano-code 流）、3 層構造で全部自前 |
| Python | [`python/`](./python) | langchain サブ実装で py-phase1 完了 | フレームワーク / アプローチ別にサブ管理（LangChain 系 / 将来 pydantic-ai / autogen / vanilla 等を並列追加） |

3 言語 × 3 アプローチで「同じ抽象化を異なるエコシステムでどう表現するか」を比較する設計。同一ユースケース（repo-reader 等）を全実装で揃えると、選択判断の引き出しが増える。

各言語ディレクトリに README と実装を置く。ルート README は言語をまたぐ一覧に留める

## 構成指針

- 同一ユースケース（Tool Calling + 会話履歴 + 知識検索）を各言語・各フレームワークで実装して横並び比較
- 各言語ディレクトリはヘキサゴナル / クリーンアーキテクチャで構成
- ドメイン層と Tool 実装は各言語ディレクトリ内で共有、SDK / フレームワーク依存は adapter 層に閉じ込める
