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

## ランタイム / ツールチェーン

各サブディレクトリは「言語ごとに乱立しがちなツールを統合した選択」を採用している。Go / Rust が言語設計時から持つ「標準ツールチェーン一発で完結」の体験を、TS と Python でも近づけるのが狙い。

| サブ | ランタイム / パッケージマネージャ | 統合範囲 | 同等の Go / Rust 機能 |
|---|---|---|---|
| `go/` | Go 標準 | `go mod` / `go build` / `go test` / `go fmt` / `go vet` | 言語同梱（参照基準） |
| `ts/` | [Bun](https://bun.sh) | `bun install` / `bun run` / `bun test` / `bun build` + TS ネイティブ実行 | `cargo` に近い体験（package + runtime + test + bundle 統合） |
| `python/langchain/` | [uv](https://github.com/astral-sh/uv) | `uv sync` / `uv run` / pyproject.toml + 高速 resolver | `cargo` に近い体験（Rust 製で resolver も速い） |

### なぜ Bun を Node.js より優先するか

Node.js エコシステムは歴史的に各層で選択が乱立してきた（`npm` / `yarn` / `pnpm`、`tsc` / `swc` / `esbuild`、`jest` / `vitest` / `mocha`、`webpack` / `vite` / `parcel`、`prettier` / `biome`）。設定ファイルが累積し、バージョン整合性の管理が煩雑になる。Bun はこれらを単一バイナリに統合し、TypeScript もネイティブで実行できる。Go / Rust が最初から持っていた標準ツールチェーン体験を TS でも実現する選択。

### なぜ uv を pip / poetry より優先するか

Python はパッケージマネージャ（`pip` / `poetry` / `pipenv`）と仮想環境ツール（`venv` / `virtualenv` / `pyenv`）が分離していて運用が分裂しがち。uv はこれらを統合し、Rust 製の高速 resolver で `poetry install` 比 10 倍以上の速度を実現する。pyproject.toml を SSoT とする現代的な書き方を維持しつつ、ツールは集約する。

## 開発環境（Devcontainer）

`.devcontainer/devcontainer.json` で開発環境をコードとして定義している。VS Code / Cursor / GitHub Codespaces で `agent/` を開くとコンテナが自動構築され、Bun / Go 1.23 / Python 3.12 + uv / GitHub CLI が揃った状態で起動する。詳細は [`.devcontainer/README.md`](./.devcontainer/README.md) を参照。

ホスト OS / 個人マシンの状態に依存しない再現性を確保し、ルート README で示した「標準ツールチェーン体験を全言語で揃える」方針を開発環境レベルでも担保する。

## 構成指針

- 同一ユースケース（Tool Calling + 会話履歴 + 知識検索）を各言語・各フレームワークで実装して横並び比較
- 各言語ディレクトリはヘキサゴナル / クリーンアーキテクチャで構成
- ドメイン層と Tool 実装は各言語ディレクトリ内で共有、SDK / フレームワーク依存は adapter 層に閉じ込める
