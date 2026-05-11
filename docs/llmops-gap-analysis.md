# LLMOps カバレッジ ギャップ分析

このリポの実装（go / ts / python）に LLMOps の運用規律をどこまで載せているかの自己評価。「動く実装はあるが、運用の規律が乗っていない」状態を可視化し、足すべき優先順位を整理する。

## 背景 — LLMOps とは

| 観点 | MLOps | LLMOps |
|---|---|---|
| 主役 | 自前で訓練したモデル | 他社の基盤モデル（Claude / GPT / Gemini） |
| 改善の単位 | データ + 重み（再学習） | prompt + コンテキスト + tool（再デプロイなし） |
| バージョン管理 | データセット + 重み + ハイパラ | prompt + RAG index + tool schema + model ID |
| 評価 | accuracy / F1 / AUC（数値で一意） | LLM-as-Judge / rubric / human eval（多次元・揺らぎ） |
| 失敗モード | overfitting / data drift | hallucination / prompt injection / context rot |
| 改善サイクル | 週〜月 | 時間〜日 |

このリポは LLMOps 寄り（外部 API を呼ぶエージェント群）。隣接する MLX + LoRA fine-tune リポが MLOps 寄り。

## 現状把握

| 実装 | 内容 | LLMOps 規律 |
|---|---|---|
| `ts/core/` | 3 プロバイダー実装（Anthropic / OpenAI / Google）+ Agent 思考ループ | observability なし、eval なし、prompt registry なし、cache 未対応 |
| `ts/agents/repo-reader/` | 最小エージェント（readFile / listFiles tool） | prompt はコード直書き、出力評価なし |
| `python/langchain/` | LangChain ベースの doc_reader / search_agent | 同上 |
| `go/` | Genkit ベースの実装 | 同上 |

`ts/core/types.ts:39-43` に `Usage { inputTokens / outputTokens / totalTokens }` の型は定義済だが、call 単位の集計・コスト計算・JSONL ログ出力は未実装。

## 不足箇所（致命度順）

### 1. Observability（致命的欠落）

ts / python 両方で token / cost / latency / tool trace の計測なし。Agent 毎のコスト分析なしでは、実装選択（cache 入れるか / streaming にするか）の意思決定ができない。

**最小構成**: `ts/cli/observability.ts` を作り、`Agent.run()` の前後で計測 → JSONL に出力。span 設計は OpenTelemetry GenAI Semantic Conventions に準拠（`gen_ai.request.model` / `gen_ai.usage.input_tokens` / `gen_ai.provider.name` / `gen_ai.operation.name`）すれば後から Langfuse / Phoenix / Datadog どれにも繋げられる。

推定 2h。

### 2. Eval（両リポで不在）

`agents/repo-reader` の出力品質を測る仕組みがゼロ。同入力で出力が変わった時に「良くなった / 悪くなった」を機械的に判定できない。

**最小構成**: `ts/agents/repo-reader/eval.ts` で golden な README を持っておき、生成結果と diff（後で BLEU / ROUGE / LLM-as-Judge に拡張）。

LLM-as-Judge を導入する時は、2026 時点のベストプラクティスに従う:
- pairwise + 両順序一致のみ valid
- rubric を criterion-separated に分解
- cross-model judge（生成と judge で別モデルを使う）

推定 3h。

### 3. Prompt Caching（簡単な勝ち筋）

`ts/core/providers/anthropic.ts:101-113` の `client.messages.create()` に `cache_control` パラメータなし。同じシステム prompt + tool 定義を毎回送っているのに cache していない。

**実装**: system / tool 定義に `ephemeral` マーク、ログで cache_hit を記録。OpenAI / Anthropic / Gemini いずれも cached input 約 90% 割引が標準で、ヒット率が直接コストに効く。

推定 1h。

### 4. Prompt Versioning（コード直書き）

`ts/agents/repo-reader/prompt.ts`、`python/agents/doc_reader/prompt.py` は文字列定数。git diff で追跡はできるが、A/B test や registry 化なし。「どの prompt 版がどのモデルで何を出力したか」のメタデータがないと eval が意味を持たない。

**実装**: `core/prompts.json` に外出し、version タグで管理。

「微小な prompt 変更で 20–50% の品質低下が起こりうる」という観測が 2026 時点で標準的に共有されており、registry + canary は思っているより重要。

推定 1h。

### 5. MLOps ⇄ LLMOps 経路なし

隣接する fine-tune リポ（MLX + LoRA）の adapter 評価は手動 5 問採点。fine-tune した adapter を agent で使った時の品質を、agent 側 eval と統一指標で見る経路がない。

**設計**: fine-tune リポ側で評価メトリクス定義 → agent 側で eval ハーネス実装 → 両者を CI で自動比較。

推定 2h。

## 学習効果順の実装優先

合計 9h で **「自分のエージェントを LLMOps 規律で運用する最小セット」** が揃う。

1. Observability logger（2h）
2. Eval harness（3h）
3. Prompt versioning（1h）
4. Caching support（1h）
5. MLOps feedback loop（2h）

## 2026 LLMOps スタック（参考）

実装時に参照するツール群:

- **Observability**: Langfuse（ClickHouse が 2026/1 に買収、勢いある）/ Arize Phoenix（OTel ネイティブ）/ Helicone / LangSmith / Datadog LLM Observability
- **Eval**: Braintrust / Promptfoo / DeepEval / Latitude / Maxim AI
- **Prompt Registry**: Braintrust / PromptLayer / Maxim AI / Langfuse Prompt Management
- **Gateway**: LiteLLM（OSS, 100+ provider）/ Portkey / Helicone Gateway
- **Routing**: small model first / cascade routing（小モデル → 信頼度低なら大モデルにエスカレーション）

## 2025 → 2026 のシフト

- **Agent eval**（multi-turn / trajectory / tool-use 軌跡）が単発 prompt eval を上回って主役化
- **tau2-bench** が voice / knowledge retrieval / policy adherence まで拡張
- trajectory metrics（全推論ステップ）と outcome metrics（最終達成）の両軸評価が必須化
- **OpenTelemetry GenAI Semantic Conventions** が事実上の標準化レイヤ
- **LLM gateway** はコスト削減ツールから governance + observability + multi-provider failover の中央集約レイヤへ昇格

## 参考

- OpenTelemetry GenAI Semantic Conventions: https://dev.to/x4nent/opentelemetry-genai-semantic-conventions-the-standard-for-llm-observability-1o2a
- Demystifying Evals for AI Agents (Anthropic): https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents
- Top 5 Prompt Versioning Tools 2026: https://www.getmaxim.ai/articles/top-5-prompt-versioning-tools-in-2026/
- Top 5 LLM Gateways 2026: https://dev.to/varshithvhegde/top-5-llm-gateways-in-2026-a-deep-dive-comparison-for-production-teams-34d2
- Prompt Caching with OpenAI, Anthropic, Google: https://www.prompthub.us/blog/prompt-caching-with-openai-anthropic-and-google-models
