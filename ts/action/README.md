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

## ループとしての状態管理（loop-engineering 方式）

Action 層は会話をまたいで動くため、状態をコンテキストウィンドウに持てない。
cobusgreyling/loop-engineering の「宣言 + 状態」分離を採用する。

| ファイル | 役割 |
|---|---|
| `LOOP.md` | ループの宣言。目的 / non-goals / 監視対象 / 段階（L1-L3）/ エスカレーション条件 / denylist |
| `STATE.md` | 耐久的な外部状態。実行前に読み、実行後に書き戻す。解決済みは剪定 |
| `loop-budget.md` | コスト上限。1 項目の最大イテレーション、1 日の自動 PR 上限、停止基準 |
| `loop-run-log.md` | 実行記録（日付 / 結果 / 検出数） |

運用原則。

- 段階展開 — L1（レポートのみ）で誤検知率を確認してから、L2（単一ファイル限定の支援修正）へ昇格する。L3（無人化）は当面やらない
- Maker / Checker 分離 — implementer が自分の作業を完了扱いにしない。検証（テスト実行）は別エージェント + worktree 隔離で行う
- 停止条件の判定は実行エージェント自身にさせず、別コンテキストで検証する
- 同一項目で 2 回連続失敗したらループを止めて Issue でエスカレーションする

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
