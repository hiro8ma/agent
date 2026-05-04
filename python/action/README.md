# Action 層

CI/CD 統合レイヤー。GitHub Actions から呼ばれるエントリ、Issue → コード修正 → PR 作成のワークフローを担う。

## 責務

- GitHub Actions ワークフロー用のエントリ実装
- Issue / PR コメントからの起動 trigger 処理
- リポへの commit / push / PR 作成
- 実行ログを Issue / PR にコメントとして post

## 依存

- CLI 層（`cli.agent`、Tool 群）
- Core 層（`core.providers`）
- `PyGithub` or `gh` CLI

## ファイル構成（予定）

```
action/
├── __init__.py
├── github.py            GitHub API ラッパ（PyGithub）
├── issue_resolver.py    Issue → エージェント実行 → PR
└── README.md
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
action/issue_resolver.py
  ↓
agents/repo_reader/runner.py を実行
  ↓
knowledge/repos/langchain.md を生成
  ↓
新規 branch + commit + PR 作成
  ↓
Issue にコメント「PR #123 を作成しました」
  ↓
人間がレビューして merge
```
