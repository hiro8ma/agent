---
title: "コーディングエージェントの設計パターン（2026 年）"
date: "2026-05-01"
tags: [agent, coding-agent, design-pattern, sandbox, tool-loop]
---

# コーディングエージェントの設計パターン（2026 年）

Claude Code / Cursor / Aider / OpenHands / SWE-agent / Cline / continue.dev / Devin など、2026 年時点のコーディングエージェント実装を横断して見えてきた共通パターンを整理する。

## 4 つの抽象化レイヤー

フレームワーク（LangChain / Vercel AI SDK / Genkit 等）に依存せず最小構成で組むと、以下の 4 層が浮かび上がる。

```
┌────────────────────────────────────┐
│ 1. LLM 抽象化（Provider SPI）       │
│    doGenerate / doStream            │
│    Anthropic / OpenAI / Google を   │
│    1 つのインタフェースで切替       │
├────────────────────────────────────┤
│ 2. Tool 定義                        │
│    name / description / parameters  │
│    needsApproval（HITL）            │
├────────────────────────────────────┤
│ 3. 思考ループ                       │
│    while (step < maxSteps)          │
│    生成 → tool 呼び出し → 結果追記  │
│    stop で終了                      │
├────────────────────────────────────┤
│ 4. サンドボックス                   │
│    namespace / VM / WASM 隔離       │
│    permission scope                 │
└────────────────────────────────────┘
```

各層は単独で差し替え可能。プロバイダ変更（Anthropic → Vertex AI）もサンドボックス変更（Docker → Firecracker）も他層に影響しない。

## Diff-based Editing

`oldText/newText` の文字列置換が unified diff より LLM 出力安定。複数マッチ時はエラー返却が定石。Claude Code の Edit tool、Aider の SEARCH/REPLACE block、nano-code の `editFile` が同型。

```ts
// 安定する
edit({ oldText: "func Foo()", newText: "func Bar()" })

// 不安定（LLM が diff 形式を間違えやすい）
applyPatch({ patch: "@@ -10,5 +10,5 @@\n-func Foo()\n+func Bar()" })
```

複数マッチ時のエラー返却は LLM に「より具体的な oldText を指定してください」と再試行させる契機になる。

## 思考ループの停止判定

3 つの条件を AND で監視:

1. `finish_reason == "stop"`（モデルが自然終了を返した）
2. `step < maxSteps`（ハードリミット、nano-code は 10）
3. tool_use がレスポンスに含まれない

このうち 1 つでも崩れれば終了。max steps 超過は「暴走防止」、tool_use 不在は「もう Tool を呼ぶ必要がない」のシグナル。

```ts
while (step < maxSteps) {
  const response = await llm.generate({ messages, tools })
  if (response.finishReason === "stop") break
  if (!response.toolCalls.length) break
  // tool 実行 → messages に追記
  step++
}
```

## Context Compaction（コンテキスト圧縮）

長時間セッションで context window を溢れさせない仕組み。代表パターン:

1. **Sliding window**: 古いターンを単純 drop（情報損失あり、最も簡単）
2. **Summarization**: 閾値超で要約モデルを呼び、古い tool_result を要約 1 メッセージに置換
3. **Hierarchical**: 直近 N ターンは verbatim、それ以前は要約
4. **Tool 結果の選別保持**: text 部分は要約、tool_call/tool_result は重要なものだけ残す

nano-code の `manageContext()` は 30k char 超で 2 段階圧縮（要約置換 → 古い順に削除）。Claude Code の auto-compact と同パターン。

トリガーは token 数 / ターン数 / コストのいずれか。要約モデルは安価なもの（gemini-2.5-flash, claude-haiku 等）を使うと費用対効果が高い。

## Repo Map

Aider が採用したパターン。tree-sitter で関数シグネチャを抽出し、token budget 内に repository 全体の構造を圧縮注入する。

```
src/
  agent.ts:
    function runLoop(messages: Message[]): Promise<Result>
    function manageContext(messages: Message[]): Message[]
  tools/
    editFile.ts:
      function editFile(path: string, oldText: string, newText: string): void
```

LLM はファイルを開く前に「どこに何があるか」を把握できる。ファイル一覧 + grep より context 効率が良い。

## File Context Window

ファイル全文を context に積まず、line range 指定で見せる（SWE-agent ACI が源流）。

```ts
view({ path: "src/agent.ts", startLine: 100, endLine: 150 })
```

scrollable file viewer として実装すると、長いファイルでも段階的に閲覧できる。

## Sub-agent / Context Isolation

メイン context を汚さずに長い調査・実行を委譲。Claude Code の Task tool、OpenHands の MultiAgent が採用。

- メインエージェント: 計画 / 結果統合
- サブエージェント: 個別タスク（独立 context）

サブの結果は要約だけメインに返す。「コンテキストが膨らむ調査」と「精密な判断」を分離できる。

## Streaming UI

`process.stdout.write(chunk)` で逐次表示。長い応答でも体感レイテンシが下がる。ConnectRPC server-streaming や SSE で同様の構成が可能。

中断/再開を考慮するなら、chunk 単位で「セッション ID + offset」を持たせて再接続時に offset 以降だけ流す設計もある。

## Sandbox 比較

| 方式 | 起動時間 | 隔離レベル | 用途 |
|---|---|---|---|
| Docker / podman | 1-3s | kernel 共有 | OpenHands / Daytona / ローカル開発 |
| bubblewrap (bwrap) | <100ms | namespace のみ | nano-code / ローカル CLI |
| Firecracker microVM | 125-200ms | kernel 分離 | E2B / Lambda / Northflank |
| WebAssembly (WASI) | <10ms | capability based | ブラウザ / Pyodide |
| Cloud Run / Modal | 数百 ms-秒 | gVisor | スケール実行 |

実務指針:
- ローカル開発: bwrap（軽量）or Docker（汎用）
- 本番マルチテナント: Firecracker（kernel 分離が必要）
- ephemeral 実行: E2B / Modal
- 永続 dev env: Daytona

## HITL（Human-in-the-loop）承認

破壊的操作（writeFile / execCommand / git push 等）には `needsApproval: true` フラグを付け、実行前にユーザー承認を要求する。

```ts
{
  name: "writeFile",
  description: "...",
  parameters: { ... },
  needsApproval: true,
  execute: async (args) => { ... }
}
```

CI 環境では `--yolo` 相当で承認をスキップする抜け道を別途用意する。

## GitHub Actions 統合

最小構成のパターン:

```yaml
on:
  workflow_dispatch:
  issues:
    types: [opened]

permissions: {}  # デフォルト最小

jobs:
  run:
    if: contains(fromJSON('["OWNER","MEMBER","COLLABORATOR"]'), github.event.issue.author_association)
    permissions:
      contents: write
      pull-requests: write
    steps:
      - uses: actions/checkout@v4
      - run: ./bin/agent
        env:
          LLM_API_KEY: ${{ secrets.LLM_API_KEY }}
          PROVIDER: ${{ vars.PROVIDER }}
          MODEL: ${{ vars.MODEL }}
```

ポイント:
- `permissions: {}` でデフォルト最小化、必要 job だけ昇格
- `author_association` で実行可能ユーザーを制限
- API key は Secrets、設定値は Variables に分離

## まとめ

「LLM 抽象化 → Tool → 思考ループ → サンドボックス」の 4 層構造を抑えると、フレームワークが裏で何をやっているかが見えてくる。Genkit / LangChain / Vercel AI SDK 等を使う場合も、各層がフレームワークのどの API に対応しているかを意識すると debug / tuning が効く。

## 参考

- nano-code: https://github.com/laiso/nano-code
- Building Effective AI Agents (Anthropic): https://www.anthropic.com/research/building-effective-agents
- Aider: https://github.com/Aider-AI/aider
- OpenHands: https://github.com/All-Hands-AI/OpenHands
- SWE-agent: https://github.com/SWE-agent/SWE-agent
- Cline: https://github.com/cline/cline
