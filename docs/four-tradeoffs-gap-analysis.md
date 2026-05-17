# エージェントシステム設計 4 トレードオフ — このリポへの実装ギャップ

エージェントベース AI システムを設計するときの 4 軸（性能 / 拡張性 / 信頼性 / コスト）に対して、このリポ（go / ts / python のエージェント実装群）がどこまで載せているかを自己評価する。

## 4 トレードオフの整理

| 軸 | 内容 | ハイブリッド方式 |
|---|---|---|
| 性能 | スピード vs 精度 | 初動で素早く概算 → 追加時間で磨き上げ（cascade routing / draft-and-verify） |
| 拡張性 | GPU/リソース最適化 | dynamic allocation + 非同期 + ハイブリッドクラウド |
| 信頼性 | ロバスト性 | フォールトトレランス + 広範なテスト + 継続監視と学習 |
| コスト | 性能 vs 費用 | 軽量モデル + 従量課金 + cache |

## 2026 の収束パターン（Web リサーチ結果）

- **性能**: Speculative Cascades (Google Research) が draft-and-verify と cascade routing を統合した標準形。OpenRouter / Martian / Not Diamond が cost-aware routing API を提供。speculative tool use は tool call を draft で先打ちする派生
- **拡張性**: 3 大ハイパースケーラで PTU vs On-Demand に収束。break-even は ~300k TPM × 8h/day（Azure PTU は約 40% 割引）。それ未満は on-demand + serverless GPU（Modal / Replicate / Beam）でバースト吸収
- **信頼性**: τ²-bench の trajectory metrics（tool 選択正答 / 引数妥当性 / step 数 / policy 遵守）がデファクト。Guardrail 3 段スタック（Llama Prompt Guard 2 86M → LlamaGuard 3 8B → NeMo Guardrails orchestrator）
- **コスト**: Prompt Caching 割引（Anthropic 90% / Gemini 75% / OpenAI 50% read）標準化。LLMLingua-2 で 2-5x 圧縮（latency 1.6-2.9x 改善）。Batching API で 50% 追加割引

**2026 の収束**: cascade routing + prompt caching + guardrail stack + OTel GenAI 計装の 4 点セット。

## このリポ (TS / Bun) の実装評価

### 1. 性能（スピード vs 精度）— ❌ ない

- `GenerateParams` に `signal?: AbortSignal` のみで AsyncIterable streaming なし
- `core/providers/*.ts` の `doGenerate` は Promise<Result> のみ、streaming メソッドなし
- 単一モデル選択のみ（`selectProvider()` で固定）、cascade routing 未実装
- Agent ループに `maxSteps` あり（`cli/agent.ts:40-84`）も単一 API 呼び出しのタイムアウトなし、signal の actual usage なし
- Tool 実行後に次の生成を同期待機（`cli/agent.ts:71-79`）、speculative tool use なし

### 2. 拡張性（リソース最適化）— ❌ ない

- Tool 呼び出しが順次実行（`cli/agent.ts:71-79` の `for (const call of toolCalls)`）、`Promise.all()` なし
- Anthropic SDK の `cache_control` オプション未使用、キャッシュヘッダなし
- `maxTokens` パラメータあり（デフォルト 4096）も累積使用量 tracking / 予算超過検知なし
- README の「Context compaction」は予定項目（`cli/README.md:22`、`manageContext.ts` 存在せず）

### 3. 信頼性（ロバスト性）— 🟡 部分的

**載っているもの**:
- `maxSteps` 制御（`cli/agent.ts:40`）
- `finishReason==stop` 判定（`cli/agent.ts:81`）
- Tool なし → 終了（`cli/agent.ts:58`）
- Tool 引数の型チェック + ファイルサイズ制限 1 MB + ディレクトリトラバーサル検知

**載っていないもの**:
- 同一 Tool 連続呼び出し検知なし、Tool 単体のタイムアウト制御なし
- `LLMApiError.retryable` プロパティあり（`core/types.ts:90-91`）も実装側で使用されない
- 構造化 error handling（Result 型）なし、silent fail あり（`cli/agent.ts:105` で error string 返却）
- モデル出力の構造検証なし、prompt injection 対策なし
- `.test.ts` ゼロ、eval ハーネスゼロ

### 4. コスト（費用最適化）— 🟡 部分的

**載っているもの**:
- `result.usage` は全 provider で返却（`anthropic/openai/google.ts`）

**載っていないもの**:
- Agent ループで token usage 集計されない（`playground/simple-call.ts` で手動出力のみ）
- USD 計算なし、price list なし
- cache hit ratio tracking なし（cache_control 未実装）
- console.error でステップ出力のみ（`cli/agent.ts:42-43`）、JSON ログ / structured logging / trace なし

## 総合評価

| トレードオフ | 判定 | コア課題 |
|---|---|---|
| 性能 | ❌ | streaming / cascade / speculative すべて未実装 |
| 拡張性 | ❌ | concurrency / cache / budget tracking / context compression なし |
| 信頼性 | 🟡 | maxSteps + finishReason あり、retries / test / guardrails なし |
| コスト | 🟡 | usage 返却のみ、USD 計算なし、observability 最小限 |

Core 層は adapter pattern でよく設計されているが、CLI 層の Agent ループが「最小限」で、生産用途には streaming / budget / observability の 3 点が必須。phase2/3 の計画と整合する。

## 実装ギャップ Top 3（学習効果順 / 推定 26h）

1. **Streaming + Concurrency** (`core/` + `cli/agent.ts`) — AsyncIterable streaming と Tool 並列実行の両立。ユーザー体験と API 効率を同時改善。**推定 8h**
2. **Token Budget Tracking + Cost Calculation** (`cli/agent.ts` + 新 `monitoring.ts`) — 累積使用量 tracking、USD 計算、予算超過 alert。cost awareness 向上。**推定 6h**
3. **Context Compression + Test Suite** (`cli/manageContext.ts` + `test/`) — Sliding window / summarization + e2e test。信頼性と拡張性の同時底上げ。**推定 12h**

これに `docs/llmops-gap-analysis.md` で挙げた Observability logger / Eval harness / Prompt caching / Prompt registry / MLOps feedback の 5 項目（9h）を重ねると、エージェント実装としての完成度が一段上がる。

## 参考

- [Speculative Cascades (Google Research)](https://research.google/blog/speculative-cascades-a-hybrid-approach-for-smarter-faster-llm-inference/)
- [Enterprise AI Platform 2026 (Internative)](https://internative.net/insights/blog/enterprise-ai-platform-comparison-vertex-bedrock-foundry-2026)
- [τ²-bench / AI Agents 2026 architecture](https://andriifurmanets.com/blogs/ai-agents-2026-practical-architecture-tools-memory-evals-guardrails)
- [Prompt Caching Guide 2026](https://tokenmix.ai/blog/prompt-caching-guide)
- [LLMLingua-2](https://llmlingua.com/llmlingua2.html)
- [Langfuse Token & Cost Tracking](https://langfuse.com/docs/observability/features/token-and-cost-tracking)
- [OTel GenAI Semantic Conventions](https://opentelemetry.io/blog/2026/genai-observability/)
