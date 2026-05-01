# CLI 層

ローカル実行環境。Agent クラス + 思考ループ + Tool 群 + サンドボックス + Context compaction を提供する。

## 責務

- `Agent` クラス（`run` / `streamRun` メソッド）
- 思考ループ（`while step < maxSteps`、`finishReason` 判定、`tool_use` 不在で終了）
- 標準 Tool 群（readFile / writeFile / editFile / listFiles / grep / execCommand / gitClone 等）
- サンドボックス（bwrap / Docker / なし、を選択可能）
- Context compaction（sliding window / summarization）

## 依存

- Core 層（`@core/types`、Provider）

## ファイル構成（予定）

```
cli/
├── agent.ts              Agent クラス + 思考ループ
├── manageContext.ts      compaction（閾値超で要約 → 削除）
├── sandbox.ts            bwrap / Docker / none
├── tools/
│   ├── readFile.ts
│   ├── writeFile.ts
│   ├── editFile.ts       diff-based editing
│   ├── listFiles.ts
│   ├── grep.ts
│   ├── execCommand.ts
│   └── gitClone.ts
└── index.ts
```

## 設計方針

- Tool には `needsApproval` フラグを持たせ、破壊的操作は HITL 承認を要求
- ループ停止判定は 3 条件 AND（`finishReason==stop` / `step<maxSteps` / `tool_use` あり）
- diff-based editing は `oldText/newText` 置換、複数マッチ時はエラー返却
- Tool 実行はサンドボックス越しが原則（local 実行は明示的に opt-in）
