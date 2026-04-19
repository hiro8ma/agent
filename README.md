# AI Agent

AI エージェントの実装を複数言語・複数フレームワークで並べて比較するモノレポ

## Role in the Ecosystem

このリポは **Tool 消費層**。AI エージェント本体（LLM 呼び出し / Tool Dispatch ループ / 会話履歴 / プロンプト管理）を各言語・各フレームワークで実装する。エージェントが呼び出す Tool（MCP サーバー）は [`../mcp/`](../mcp/) を参照

## Implementations

| 言語 | ディレクトリ | 状態 |
|---|---|---|
| Go | [`go/`](./go) | 実装済み（genai / Genkit / ADK） |
| TypeScript | `typescript/` | 追加予定 |
| Python | `python/` | 追加予定 |

各言語ディレクトリに README と実装を置く。ルート README は言語をまたぐ一覧に留める

## 構成指針

- 同一ユースケース（Tool Calling + 会話履歴 + 知識検索）を各言語・各フレームワークで実装して横並び比較
- 各言語ディレクトリはヘキサゴナル / クリーンアーキテクチャで構成
- ドメイン層と Tool 実装は各言語ディレクトリ内で共有、SDK / フレームワーク依存は adapter 層に閉じ込める
