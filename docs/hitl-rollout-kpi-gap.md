# HITL・段階展開・KPI 監視 — このリポへの実装ギャップ

## TL;DR

このリポを **Human-in-the-Loop / Progressive Rollout / KPI 監視ダッシュボード**の 3 観点で評価。HITL ❌ / Progressive Rollout 🟡 部分的 / KPI 監視 ❌。本回固有の実装ギャップは合計 **65h**。既存 docs（four-tradeoffs 26h + architecture-and-eval 112h + llmops 9h）と合わせて、Production-ready なエージェント実装に必要な積み重ねが明確化される。

## 教科書整理

### Human-in-the-Loop (HITL)

- 専門家レビュー、実運用に近い環境での検証
- フィードバック収集 → 正解データ化 → 評価ループ

### 実世界テストと継続的サイクル

1. 段階的にデプロイ（小規模 → 拡大）
2. 振る舞いを監視（KPI tracking）
3. ユーザーフィードバック収集
4. 洞察に基づき反復

### scoped use case

- 「最も派手な」ではなく **「毎週の業務摩擦が大きい high-frequency / low-risk」**
- narrow scope + 高入力構造 + 低エラー影響（classification / extraction / summarization / routing / flagging）

## 2026 の収束パターン（Web リサーチ結果）

### HITL

- RLHF の PPO は **DPO / KTO / GRPO / DAPO** に置換
- **RLTHF（RLAIF + 人間ハイブリッド）** が主流、**人間ラベルを 6% に圧縮**して同等精度
- Uncertainty / Query-by-Committee で「人間に聞くべきケース」を絞る
- UI は **thumbs → 詳細フィードバック → カテゴリ分類の 3 段階**

### Shadow / Canary

- **Shadow mode**: 本番 traffic を mirror し LLM-as-Judge で出力比較（factuality / tone / format / token / latency）
- **Canary**: prompt label (prod-a / prod-b) で段階配信、品質しきい値で promote
- Langfuse は version label + ランダム配信、Helicone / Portkey は gateway 型

### 運用 KPI / 4 ダッシュ

- **OpenTelemetry GenAI Semantic Conventions** が標準
- Cost / Quality / Latency / Tool 実行の 4 軸
- `invoke_agent` → `chat` → `execute_tool` の span tree

## このリポ (TS/Bun) の実装評価

### 1. Human-in-the-Loop (HITL) — ❌ ない

**根拠**:
- `core/types.ts:16` に `needsApproval?: boolean` 定義のみで実装なし
- `cli/agent.ts:71-79` Tool 実行ループにガード機構なし、実行前のユーザー確認パイプラインなし
- フィードバック受付（thumbs / コメント / カテゴリ）の型・永続化機構ゼロ
- golden dataset 生成パイプラインなし
- runtime 介入（停止 → 人間修正 → 再開）なし

**優先 Top 2 ギャップ**:
1. **Tool 承認ゲート + フィードバック収集** (`cli/agent.ts` + 新 `cli/feedback.ts` / **12h**) — `needsApproval` フロー実装、approval dialog、JSON feedback log（vote / comment / category）永続化
2. **フィードバック → golden dataset pipeline** (`agents/repo-reader/eval.ts` + `cli/feedbackIngest.ts` / **8h**) — 承認済み出力を golden と型付け、LLM-as-Judge で自動ラベリング、eval harness 統合

### 2. Progressive Rollout — 🟡 部分的

**根拠**:
- `factory.ts:11-32` で `LLM_PROVIDER` / `LLM_MODEL` env vars のみ対応、環境別設定（dev/staging/prod）の構造なし
- `action/README.md:27-33` に stage 分離は記述のみ、実装ファイルなし
- feature flag / kill switch なし
- shadow mode（本番 traffic mirror + 出力比較）の概念なし
- canary % 段階展開の仕組みなし
- A/B test（prompt 版 / model 版）の対照実験機構なし

**既存 doc 参照**: `docs/architecture-and-eval-gap.md` の「stage / feature flag 機構（~10h）」と一部重複

**優先 Top 2 ギャップ**:
1. **環境別設定 + feature flag framework** (`core/config.ts` 新規 + `.env` parsing / **8h**) — `ENV=dev|staging|prod`、feature flag enum（`MULTI_AGENT=bool` / `STREAMING=bool` / `FEEDBACK_LOOP=bool`）、env-specific override
2. **Canary + shadow mode infra** (`cli/canary.ts` + `cli/shadowRunner.ts` / **15h**) — %-based traffic routing、mirror output capture、自動 diff 出力（JSON format）、divergence threshold で rollback trigger

### 3. KPI 監視ダッシュボード — ❌ ない

**根拠**:
- Cost: `core/providers/*.ts` で `usage` 返却のみ（`anthropic.ts:129-137`）、token 集計 / USD 計算 / cache hit ratio なし
- Quality: 出力構造検証なし、`finishReason` 集計なし（`cli/agent.ts:81` でチェック後に捨てる）、retries 率ゼロ
- Latency: TTFT / TPOT / e2e 計測なし、console.error でステップ出力のみ（`cli/agent.ts:42-43`）
- Tool 実行: tool 選択正答率（golden 比較）なし、引数妥当性検証なし
- ログ: JSON 構造化ログなし、OTel format なし、dashboard connector なし

**既存 doc 参照**: `docs/llmops-gap-analysis.md` の「Observability logger（2h）」と一部重複、本回は KPI dashboard 全体に拡張

**優先 Top 2 ギャップ**:
1. **KPI 計測 + JSON ログ backend** (`cli/metrics.ts` + `cli/observability.ts` / **10h**) — Agent ループで span 開始/終了、token / latency / tool 数を **OpenTelemetry GenAI Semantic Conventions** で記録、JSONL file output、Langfuse / Datadog 接続用の HTTP exporter
2. **KPI dashboard + regression detection** (`__tests__/dashboard.ts` + metrics aggregator / **12h**) — Cost (USD + cache ratio) / Quality (stop reason distribution / error rate) / Latency (p50/p95/p99) / Tool accuracy (golden vs output)、CLI で JSON → HTML report 生成、CI で前回比 threshold check

## 総合評価

| 観点 | 判定 | 根拠（file:line） | Top 2 ギャップ（推定工数） |
|---|---|---|---|
| HITL | ❌ | `core/types.ts:16` 定義のみ、`cli/agent.ts:71-79` 実装なし | ① Tool 承認 gate + feedback 収集 (12h)、② feedback → golden pipeline (8h) |
| Progressive Rollout | 🟡 部分的 | `factory.ts:11-32` env-based model select のみ、`action/README.md:27-33` 記述のみ | ① env 別 config + feature flag (8h)、② canary + shadow (15h) |
| KPI 監視 | ❌ | `anthropic.ts:129-137` usage 返却のみ、`cli/agent.ts:42-43` console.error のみ | ① metrics + OTel logging (10h)、② dashboard + regression (12h) |

## 本回固有の実装ギャップ（既存 doc 非重複）

- HITL 全体: **20h**
- Progressive Rollout 全体: **23h**
- KPI dashboard: **22h**
- **合計 65h**

既存 3 docs（four-tradeoffs 26h + architecture-and-eval 112h + llmops 9h）に追加すると、Production-ready なエージェント実装に必要な積み重ねが明確化される。

## 参考

- [IntuitionLabs Active Learning HITL](https://intuitionlabs.ai/articles/active-learning-hitl-llms)
- [llm-stats Post-Training 2026 (DPO / KTO / GRPO / DAPO)](https://llm-stats.com/blog/research/post-training-techniques-2026)
- [TianPan LLM Gradual Rollout](https://tianpan.co/blog/2026-04-09-llm-gradual-rollout-shadow-canary-ab-testing)
- [Langfuse A/B Testing](https://langfuse.com/docs/prompt-management/features/a-b-testing)
- [OpenTelemetry GenAI Observability 2026](https://opentelemetry.io/blog/2026/genai-observability/)
- [Datadog LLM Observability](https://docs.datadoghq.com/llm_observability/)
- [OpenObserve OTel for LLMs 2026](https://openobserve.ai/blog/opentelemetry-for-llms/)
- [linesNcircles Enterprise AI Use Cases 2026](https://linesncircles.com/Blog/Enterprise/Enterprise_AI_Use_Cases_2026)
