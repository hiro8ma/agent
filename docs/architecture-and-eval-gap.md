# エージェントアーキテクチャ・反復設計・評価戦略 — このリポへの実装ギャップ

## TL;DR

このリポ（go / ts / python のエージェント実装群）を**シングル vs マルチエージェント**、**反復設計の MVP 出口条件**、**評価戦略**の 3 観点で評価。TS/Bun 実装は **MVP 動作（phase1）は完、ただしテスト 0 / 評価 0 / マルチ化準備は 🟡 部分的**。Phase 2 の優先は **MVP 出口条件（テスト + feature flag）→ 評価戦略（e2e + LLM-as-Judge）→ マルチ化（Phase 3）** の順。

## 教科書整理

### シングル vs マルチエージェント

| 観点 | シングル | マルチ |
|---|---|---|
| 構造 | 単一エージェントが全タスク | 複数エージェントが協働・並列・調整 |
| メリット | 設計・開発・デプロイ容易 | 専門化 / 並列 / 拡張性 / 冗長性 |
| 適用 | FAQ / 注文追跡 / 単純タスク | 金融 / セキュリティ / 複雑分散 |
| 罠 | スケール限界 | オーケストレーション複雑、トークン爆発 |

### 反復設計

1. 素早くプロトタイプ（コア機能で「動作して価値を提供」）
2. テストとフィードバック収集
3. 改良と反復

### 評価戦略

- 網羅的テスト（正常系 + エッジケース + 異常系）
- 個別テスト（各スキル / モジュール単位）

## 2026 の収束パターン

### マルチエージェント

- **主要 framework**: LangGraph v0.4 / CrewAI / AutoGen 1.0 GA / OpenAI Agents SDK（Swarm 後継）
- **Anthropic Multi-agent Research System**: LeadResearcher + ephemeral Subagent 並列起動。**Opus 4 lead + Sonnet 4 sub が単体 Opus 4 を 90.2% 上回った**
- **Finance-Agent 研究**: 並列タスクで +80.9%、逐次計画で -39〜70% 劣化
- **peer mesh の罠**: O(n²) 通信爆発、infinite loop
- **2026 標準**: 単体 agent が full context 保持、必要時に ephemeral subagent を spawn して要約だけ返すハイブリッド型、**実用上 3-4 agent が上限**

### 反復設計と MVP

- Anthropic "Building Effective Agents" の方針は 2026 でも不変
- **Three-Agent Harness（2026/4）**: planning / generation / evaluation を分離
- **Progressive rollout**: shadow mode → canary → 全展開を **eval gate で制御**

### 評価戦略

- **網羅 bench**: τ²-bench / SWE-bench Verified（**Opus 4.7 が 87.6%**）/ GAIA / AgentBench / MultiAgentBench
- **個別 eval**: DeepEval が 50+ metric
- **LLM-as-Judge**: pairwise + 両順序一致 + 第三 model tie-break + criterion-separated rubric + IRT 校正
- **Guardrails 二段**: Prompt Guard 2 (86M, 20-50ms) → LlamaGuard 3 8B (INT4 4GB / 20ms p50)
- **市場**: Promptfoo が OpenAI に 86M USD で買収、Langfuse は 20k stars

## このリポ (TS/Bun) の実装評価

### 1. マルチエージェント化への準備度 — 🟡 部分的

**根拠**:
- `core/types.ts:68-70` — `LanguageModel` インタフェースは再利用可能設計 ✅
- `cli/agent.ts:16-29` — Agent クラスは単一ループのみ、複数並列実行の仕組みなし ❌
- `cli/agent.ts:71-79` — Tool 実行は順序実行（`for...of`）、並列化なし ❌
- `README.md:17-38` — 3 層アーキテクチャは独立性を保持 ✅
- Agent 間メッセージパッシング / context window 分離: 設計されていない ❌

**優先 Top 2 ギャップ**:
1. **並列 Tool 実行・Agent 実行基盤** (`cli/agent.ts` / `~20h`) — `Promise.all` + Tool グループ化、Agent pool、message bus、sub-agent パターン（親が子を呼ぶ）の型定義
2. **Agent 間 context 分離・メッセージング** (`core/types.ts` 拡張 / `~15h`) — `agentId` / `contextWindow` per-agent、shared vs isolated state、IPC-like message routing

### 2. 反復設計の MVP 出口条件 — 🟡 部分的

**根拠**:
- `package.json:10` — `bun test` はあるが **test file 0 個** ❌
- `README.md:108-112` — phase1/2/3 の段階定義あり ✅
- `bin/repo-reader.ts` / `agents/repo-reader/runner.ts` — 単一エージェント実装のみ（phase1 完）✅
- `action/README.md:27-33` — stage 分離（dev / staging / prod）は記述あり、**実装なし** ❌
- feature flag / toggle: 未実装 ❌

**優先 Top 2 ギャップ**:
1. **テスト基盤・golden dataset** (`__tests__/` 新規 / `~25h`) — Tool 単体テスト、Provider mock、repo-reader e2e テスト（入力 repo → 期待出力 Markdown の fixture）、LLM-as-Judge 評価スクリプト
2. **stage / feature flag 機構** (`core/config.ts` 新規 + env 別設定 / `~10h`) — `DEV/STAGING/PROD` env、feature flag enum（`MULTI_AGENT=true/false`、`STREAMING=true/false`）、段階的ロールアウト検証

### 3. 評価戦略の実装 — ❌ ない

**根拠**:
- e2e テスト: なし ❌
- 単体テスト: なし ❌
- 実行時間・token 計測: `core/types.ts:39-43` に `Usage` 型あるが計測ロジックなし ❌
- golden dataset: なし ❌
- LLM-as-Judge 評価: なし ❌

**優先 Top 2 ギャップ**:
1. **e2e + 評価フレームワーク** (`__tests__/eval/` 新規 / `~30h`) — repo-reader の入力（公開 OSS 3-5 個）× 期待出力 fixture、`claude-opus` as judge で `output.completeness` / `output.accuracy` / `architecture clarity` の 3 軸スコアリング、token 数・API cost 集計
2. **ベンチマーク計測・レポート** (`cli/benchmark.ts` + `__tests__/perf/` / `~12h`) — Agent 実行時間分布、tool 呼び出し数分布、model ごとの token efficiency（Haiku vs Sonnet）、回帰検出（前回比）

## 総合評価

| 観点 | 判定 | コア課題 |
|---|---|---|
| マルチ化準備度 | 🟡 部分的 | Provider 抽象は良、並列実行 / context 分離 / sub-agent パターンなし |
| MVP 出口条件 | 🟡 部分的 | phase 定義あり、test 0 / feature flag なし |
| 評価戦略 | ❌ ない | e2e / 単体 / golden / LLM-as-Judge / ベンチマーク全てゼロ |

## 公開前必須対応（推定合計 112h）

| 観点 | ギャップ | 工数 | 優先度 |
|---|---|---|---|
| マルチ化 | 並列実行基盤 | 20h | 後（Phase 2-3） |
| マルチ化 | Agent context 分離 | 15h | 後 |
| MVP | テスト基盤 + golden | 25h | **高** |
| MVP | stage / feature flag | 10h | **高** |
| 評価 | e2e + LLM-as-Judge | 30h | **高** |
| 評価 | ベンチマーク計測 | 12h | **高** |

**Phase 1 完了（repo-reader 単体動作）の次は、MVP 出口条件（テスト + feature flag）→ 評価戦略（e2e + 計測）を優先。マルチ化は Phase 3 の関心**。これに `docs/four-tradeoffs-gap-analysis.md`（性能 / 拡張性 / 信頼性 / コスト 26h）と `docs/llmops-gap-analysis.md`（Observability / Eval / Caching / Versioning / MLOps feedback 9h）を重ねると、エージェント実装としての完成度が一段上がる。

## 参考

- [Anthropic Multi-agent Research System](https://www.anthropic.com/engineering/multi-agent-research-system)
- [Anthropic Building Effective AI Agents](https://www.anthropic.com/engineering/building-effective-agents)
- [Three-Agent Harness (InfoQ)](https://www.infoq.com/news/2026/04/anthropic-three-agent-harness-ai/)
- [Multi-Agent in Production 2026](https://medium.com/@Micheal-Lanham/multi-agent-in-production-in-2026-what-actually-survived-f86de8bb1cd1)
- [Framework Comparison 2026](https://o-mega.ai/articles/langgraph-vs-crewai-vs-autogen-top-10-agent-frameworks-2026)
- [τ²-bench](https://github.com/sierra-research/tau2-bench)
- [AI Agent Benchmarks 2026](https://rapidclaw.dev/blog/ai-agent-benchmarks-2026)
- [Rubric-Based LLM-as-Judge](https://medium.com/@adnanmasood/rubric-based-evals-llm-as-a-judge-methodologies-and-empirical-validation-in-domain-context-71936b989e80)
- [DeepEval Alternatives 2026](https://www.braintrust.dev/articles/deepeval-alternatives-2026)
