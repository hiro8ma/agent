# Devcontainer

agent モノレポの開発環境を [Dev Container Specification](https://containers.dev/) に従って定義する。

## 使い方

VS Code / Cursor で `agent/` を開くと「Reopen in Container」のプロンプトが表示される。受諾すると Docker イメージをビルドし、コンテナの中でエディタが起動する。CLI から起動する場合は `devcontainer up --workspace-folder .` を使う。GitHub Codespaces からも同設定で起動できる。

## 含まれるもの

| ツール | 用途 |
|---|---|
| Bun | `ts/` のランタイム / パッケージマネージャ |
| Go 1.23 | `go/` のビルド |
| Python 3.12 + uv | `python/langchain/` の依存管理 |
| Node.js LTS | TS ツール互換性確保用 |
| GitHub CLI | PR / Issue 操作 |
| zsh + common-utils | シェル環境 |

## postCreateCommand

`.devcontainer/post-create.sh` がコンテナ初回起動時に実行される。

- `ts/` で `bun install`
- `python/langchain/` で `uv sync`
- `go/` で `go mod download`

## 設計思想

ルートの README で示している「Go / Rust の標準ツールチェーン体験を TS（Bun）/ Python（uv）でも再現する」方針を、開発環境レベルで保証する。リポを開いた人全員が同じバージョンの Bun / Go / uv で開発する。

API キー（ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY）はコンテナの環境変数として注入する。Codespaces なら Secrets、ローカルなら `.devcontainer/devcontainer.env`（`runArgs` で `--env-file` 指定）等で渡す。シークレットを `devcontainer.json` 内に直書きしない。
