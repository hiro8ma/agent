# agents/

個別エージェントの実装を置くディレクトリ。各エージェントは独立したサブディレクトリに閉じ、Core / CLI / Action 層には変更を加えずに追加できる。

## エージェントの作法

新しいエージェント `{name}` を追加する手順:

1. `agents/{name}/` ディレクトリを作る
2. 以下のファイルを置く

```
agents/{name}/
├── prompt.ts       system prompt（定数 export）
├── runner.ts       本体（CLI からも Action からも呼ばれる）
└── output.ts       結果の整形・書き出し（任意）
```

3. `bin/{name}.ts` に CLI エントリを置く
4. （CI 統合する場合）`bin/{name}-action.ts` に Action エントリを置く

## runner.ts の構造

```ts
import { Agent } from "@cli/agent"
import { readFile, listFiles } from "@cli/tools"
import { selectProvider } from "@core/providers/factory"
import { SYSTEM_PROMPT } from "./prompt"

export type RunInput = { /* エージェント固有の入力 */ }
export type RunOutput = { /* エージェント固有の出力 */ }

export async function run(input: RunInput): Promise<RunOutput> {
  const agent = new Agent({
    provider: selectProvider(),
    systemPrompt: SYSTEM_PROMPT,
    tools: [readFile, listFiles],
    maxSteps: 20,
  })
  const result = await agent.run(formatInputAsMessage(input))
  return parseOutput(result)
}
```

CLI / Action の両方から `run` 関数を呼び出す。エントリ側は引数パースと結果出力だけを担う。

## 計画中のエージェント

| 名前 | 用途 | 状態 |
|---|---|---|
| repo-reader | OSS リポを読み解いて要約を生成 | 設計中（最初の実装対象） |
| dev-digest | URL / PDF を要約してナレッジ化 | アイデア段階 |
| memory-curator | `.claude/memory/` の重複検出・整理 | アイデア段階 |
| interview-coach | 面接対策の Q&A 生成 | アイデア段階 |
