# Devcontainer

agent モノレポの開発環境を [Dev Container Specification](https://containers.dev/) に従って定義する。`Dockerfile` をビルドしてコンテナを起動する方式（image + features 方式ではなく自前 Dockerfile 方式）。

## 含まれるもの

| ツール | バージョン | 用途 |
|---|---|---|
| Bun | 1.3.11 | `ts/` のランタイム / パッケージマネージャ |
| Go | 1.23.4 | `go/` のビルド |
| Python | 3.12 | `python/langchain/` のランタイム |
| uv | 最新 | `python/langchain/` の依存管理 |
| GitHub CLI | apt stable | PR / Issue 操作 |
| git, curl, sudo, build-essential | apt | 基本ツール |

ベースイメージは Ubuntu 24.04 LTS、非 root ユーザー `vscode`（sudoers 登録済）で実行。

バージョン pinning は `Dockerfile` の `ARG` で集約。更新時は `BUN_VERSION` / `GO_VERSION` を変えて再ビルドする。

## 使い方

### Dev Container CLI（VS Code を使わない場合）

VS Code に依存せずコンテナ起動 + コマンド実行できる公式 CLI。Codespaces 内部でも使われている。

```bash
# CLI を install（Bun か npm のどちらでも可）
bun install -g @devcontainers/cli
# または
npm install -g @devcontainers/cli

# コンテナをビルド + 起動
devcontainer up --workspace-folder .

# コンテナ内でコマンド実行
devcontainer exec --workspace-folder . bun --version
devcontainer exec --workspace-folder . go version
devcontainer exec --workspace-folder . uv --version

# コンテナ内でシェル
devcontainer exec --workspace-folder . bash
```

### VS Code / Cursor

`agent/` を開くと「Reopen in Container」のプロンプトが表示される。受諾すると Docker イメージをビルドし、コンテナの中でエディタが起動する。Cursor / Windsurf も VS Code フォークなので Dev Containers 拡張がそのまま動く。

### JetBrains Gateway / IntelliJ

JetBrains 系（IntelliJ IDEA / GoLand / PyCharm）は「Dev Containers」機能でこの設定を読み込んで起動できる。

### GitHub Codespaces

リポを Codespaces で開けば自動でこの設定が使われる。クラウド上でブラウザベースで動く。

## postCreateCommand

`.devcontainer/post-create.sh` がコンテナ初回起動時に実行される。

- `ts/` で `bun install`
- `python/langchain/` で `uv sync`
- `go/` で `go mod download`

## 設計思想

ルート README の「Go / Rust の標準ツールチェーン体験を TS（Bun）/ Python（uv）でも再現する」方針を、開発環境レベルで保証する。リポを開いた人全員が同じバージョンの Bun / Go / uv で開発する。

API キー（`ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GOOGLE_API_KEY`）はコンテナの環境変数として注入する。Codespaces なら Secrets、ローカルなら `.devcontainer/devcontainer.env`（`runArgs` で `--env-file` 指定）等で渡す。シークレットを `devcontainer.json` 内に直書きしない。

## なぜ Dockerfile 方式か（image + features ではなく）

- **透明性**: Dockerfile 自体が SSoT、apt パッケージ / バージョンが一目で追える
- **再現性**: features の中身や挙動に依存せず、ベースイメージとインストール手順を明示的に管理
- **ビルド時間**: features の組み合わせは順次適用されるため layer 数が増えがち、自前 Dockerfile の方が層を最適化しやすい
- **学習価値**: 「コンテナで開発環境を作る」原理を Dockerfile レベルで把握できる
