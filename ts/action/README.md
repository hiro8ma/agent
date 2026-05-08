# Action 層

CI/CD 統合レイヤー。GitHub Actions から呼ばれるエントリポイント、Issue → コード修正 → PR 作成のワークフローを担う。

## 責務

- GitHub Actions ワークフロー用のエントリ実装
- Issue / PR コメントからの起動 trigger 処理
- リポへの commit / push / PR 作成
- 実行ログを Issue / PR にコメントとして post

## 依存

- CLI 層（`@cli/agent`、Tool 群）
- Core 層（`@core/types`）

## ファイル構成（予定）

```
action/
├── github.ts            GitHub API ラッパ（octokit）
├── issueResolver.ts     Issue → エージェント実行 → PR
├── commentReporter.ts   進捗を Issue / PR にコメント
└── index.ts
```

## 設計方針

- `permissions: {}` をデフォルトとし、必要 job だけ昇格
- `author_association` で実行可能ユーザーを `OWNER` / `MEMBER` / `COLLABORATOR` に制限
- API key は GitHub Secrets、Provider 設定は GitHub Variables に分離
- 実行結果（生成 PR URL / 失敗理由）は Issue / PR コメントとして可視化
- merge は人間に残す（auto-merge は禁止）

## ワークフロー例

```
Issue: "[repo-reader] LangChain を読んで"
  ↓
GitHub Actions trigger（issues.opened）
  ↓
action/issueResolver.ts
  ↓
agents/repo-reader/runner.ts を実行
  ↓
knowledge/repos/langchain.md を生成
  ↓
新規 branch + commit + PR 作成
  ↓
Issue にコメント「PR #123 を作成しました」
  ↓
人間がレビューして merge
```

## GitHub Actions ワークフロー例

[`oven-sh/setup-bun`](https://github.com/oven-sh/setup-bun)（Bun 公式 Action）で runner に Bun を配置し、`bun run bin/*` を直接実行する。トランスパイル不要で TS をそのまま動かす。

action 層の TS 実装が揃った時点で `.github/workflows/repo-reader.yml` として配置する想定。現状は形だけのリファレンス。

```yaml
# .github/workflows/repo-reader.yml（案、未配置）
name: repo-reader

on:
  issues:
    types: [opened, labeled]

permissions: {}

jobs:
  run:
    if: contains(github.event.issue.labels.*.name, 'repo-reader')
    runs-on: ubuntu-latest
    permissions:
      contents: write
      issues: write
      pull-requests: write
    steps:
      - uses: actions/checkout@v4

      - uses: oven-sh/setup-bun@v2
        with:
          bun-version: latest

      - run: bun install

      - name: Run repo-reader agent
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: bun run bin/repo-reader.ts

      - name: Commit + PR
        env:
          GH_TOKEN: ${{ github.token }}
        run: bun run bin/action.ts ${{ github.event.issue.number }}
```

### 注目点

| 行 | 意味 |
|---|---|
| `permissions: {}`（top） | デフォルト最小権限。job 単位で昇格させる GitHub Actions のベストプラクティス |
| `if: contains(...)` | label `repo-reader` がついた issue だけで発火、任意 issue で起動しない |
| `oven-sh/setup-bun@v2` | バージョン pinning は SHA 推奨だが、内部 OSS 用途では `@v2` で実用十分 |
| `bun install` 直接 | `npm ci` 不要、Bun が `bun.lockb` まで扱う |
| `bun run bin/*.ts` | TS をトランスパイルなしで直接実行。Node.js なら `tsx` / `ts-node` が必要 |
| `GH_TOKEN: github.token` | `gh` CLI を使う場合の標準パターン、scoped token を使い回さない |
