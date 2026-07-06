# LLMOps 運用モデル

この文書は、`agent/` の各エージェント実装を本番運用に近づけるための運用モデルを定義する。

既存の `docs/llmops-gap-analysis.md` は「何が足りないか」を示す文書である。
この文書は「どう運用するか」を示す文書である。

## 目的

LLMOpsの目的は、LLMアプリケーションを以下の状態にすることである。

- 失敗を観測できる
- 品質を評価できる
- コストを説明できる
- セキュリティリスクを検知できる
- 変更をrollbackできる
- 本番ログを次の評価データへ戻せる

## MLOpsとの違い

| 観点 | MLOps | LLMOps |
|---|---|---|
| 主な成果物 | 学習データ、特徴量、モデル重み | prompt、tool schema、RAG index、eval dataset |
| 改善単位 | 再学習、特徴量追加 | prompt変更、retrieval改善、tool設計、model routing |
| 評価 | accuracy、F1、AUC | groundedness、faithfulness、tool trajectory、rubric |
| リスク | data drift、concept drift、overfitting | hallucination、prompt injection、data leakage、excessive agency |
| 変更速度 | 週から月 | 時間から日 |

`agent/` は基本的にLLMOps寄りである。
ただし、embedding model、reranker、fine-tuned adapterを扱う箇所はMLOpsの規律が必要になる。

## 運用ループ

```text
Ideation
  -> Experiment
  -> Evaluation
  -> Deployment
  -> Monitoring
  -> Improvement
  -> Experiment
```

| ループ | やること | 成果物 |
|---|---|---|
| Ideation | 何を解くかを決める | ユースケース、リスク分類、成功指標 |
| Experiment | prompt / RAG / tool / modelを試す | PoC、eval結果、比較表 |
| Evaluation | 良くなったか測る | golden dataset、rubric、judge calibration |
| Deployment | 安全に出す | canary、rollback plan、versioned config |
| Monitoring | 本番で壊れていないか見る | traces、metrics、logs、feedback |
| Improvement | 本番失敗を次のテストに戻す | incident review、golden昇格、改善タスク |

## チーム責務

| 役割 | 責務 |
|---|---|
| PM | ユースケース、成功指標、リスク許容度を決める |
| AIエンジニア | prompt、tool、agent loop、RAG、PoCを作る |
| LLMOpsエンジニア | deploy、monitoring、eval、cost、incident responseを設計する |
| データエンジニア | knowledge source、ETL、embedding index、データ品質を管理する |
| ML / NLP担当 | eval設計、judge校正、reranker、fine-tuningを担う |
| セキュリティ担当 | prompt injection、tool権限、data leakage、red teamingを扱う |
| QA / 評価担当 | golden dataset、human eval、rubric、回帰評価を管理する |

小さいチームでは兼務してよい。
ただし、責務は混ぜない。

## SLO

最初に置くSLOは次。

| SLO | 初期目標 | 見る場所 |
|---|---:|---|
| Availability | 99.5% | API / CLI実行成功率 |
| End-to-end latency P95 | 10s未満 | trace |
| Tool success rate | 98%以上 | tool span |
| Cost per successful task | ユースケースごとに上限設定 | usage log |
| Eval pass rate | 90%以上 | offline eval |
| Safety block false negative | 0件を目標 | red team / online review |

PoC段階では厳密なSLOより、測れる状態にすることを優先する。

## Severity

| Severity | 例 | 対応 |
|---|---|---|
| P1 | 機密情報漏えい、危険なtool実行、広範囲の障害 | 即時停止、キー無効化、incident commanderを立てる |
| P2 | 回答品質の大幅劣化、cost runaway、主要tool停止 | fallback、rollback、当日中に修正 |
| P3 | 一部ユースケースの失敗率上昇、特定toolの遅延 | チケット化し、次スプリントで対応 |
| P4 | prompt改善案、eval追加、ログ項目追加 | backlogへ追加 |

LLM特有のP1は、従来の5xxだけでは検知できない。
prompt injection、data leakage、excessive agencyを別に見る。

## ダッシュボード

### Reliability

- request count
- success / failure
- retry count
- fallback count
- timeout
- provider error rate

### Cost

- input tokens
- output tokens
- cached input tokens
- cost per task
- model別コスト
- user / agent別コスト

### Quality

- offline eval pass rate
- groundedness
- answer relevance
- tool trajectory match
- human thumbs up / down
- re-ask rate

### Security

- prompt injection検知数
- blocked request数
- tool permission denial数
- PII redaction件数
- suspicious tool chain

### Data / RAG

- retrieval hit rate
- context precision
- context recall
- embedding model version
- index version
- stale document ratio

## 評価の三層

| 層 | タイミング | 目的 |
|---|---|---|
| Offline eval | PR / release前 | 既知ケースの回帰を防ぐ |
| Online eval | 本番実行後 | 未知の失敗、drift、ユーザー不満を検出する |
| Guardrails | 応答前 | 危険出力、機密漏えい、権限外tool実行を止める |

`agent/` では、まず `repo-reader` の小さいgolden datasetから始める。
本番ログに相当するCLI実行ログを保存し、失敗例をgoldenへ昇格させる。

## Trace設計

OpenTelemetry GenAI semantic conventionsへ寄せる。
最小属性は次。

```text
gen_ai.provider.name
gen_ai.request.model
gen_ai.operation.name
gen_ai.usage.input_tokens
gen_ai.usage.output_tokens
gen_ai.usage.cached_input_tokens
agent.name
tool.name
tool.success
```

`agent/ts` では、最初はOTel SDKを入れずJSONL traceでよい。
属性名だけOTelに合わせる。
後からLangfuse、Phoenix、Datadog、Cloud Traceへ流せる。

## リリースゲート

prompt、tool schema、model config、RAG設定を変える場合は、次を満たす。

1. 変更理由がある
2. versionが付いている
3. offline evalが通る
4. cost見積もりがある
5. rollback方法がある
6. P1リスクがない

## インシデント対応

LLMインシデントでは、アプリログだけでは足りない。
次を残す。

- prompt version
- model id
- tool schema version
- RAG index version
- retrieved document ids
- tool calls
- finish reason
- usage tokens
- safety / guardrail decision

ただし、ユーザー入力・中間生成・tool引数には機密が含まれる。
ログへ出す前にredactionする。

## 実装ロードマップ

| 優先 | 実装 | 目的 |
|---|---|---|
| 1 | `ts/cli/observability.ts` | JSONL trace logger |
| 2 | `ts/agents/repo-reader/eval.ts` | golden dataset評価 |
| 3 | prompt / tool schema / model configのversion記録 | regression調査 |
| 4 | provider層のcache / retry / fallback | costと信頼性 |
| 5 | guardrail / security test | P1事故防止 |
| 6 | OTel exporter | 外部observability連携 |

## 参照

- OpenTelemetry GenAI semantic conventions: https://github.com/open-telemetry/semantic-conventions-genai
- Google Cloud Gen AI evaluation service: https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/evaluation-overview
- OWASP Top 10 for LLM Applications: https://owasp.org/www-project-top-10-for-large-language-model-applications/
- LangSmith Observability: https://docs.langchain.com/langsmith/observability
