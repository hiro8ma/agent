# AI ライフサイクル・HITL・データ運用 — このリポと隣接 mcp の実装ギャップ

## TL;DR

agent/ts + agent/python + 隣接 mcp 群を **「データ収集 → ラベリング → HITL 3 層 → KPI ダッシュボード」** の観点で評価。**mcp/ai_knowledge/tracing.py の JSONL + Langfuse dual-write が既に機能**しているが、agent 側には feedback layer / eval loop / dashboard が皆無。**Feedback MCP server (10h) + Dashboard + Eval loop (18h) + Integration (8h) = 36h で 2 週間スプリント完結**。

## 教科書整理

### AI ライフサイクル

1. プロジェクトスコーピング (PRD → 技術境界)
2. データ収集 (多様なソース)
3. 検証・前処理 (ラベリング = 価値の源泉)
4. デプロイ + リアルタイム FB ループ

### HITL 3 層

- アノテーター: データ修正・ラベリング
- 専門家評価: バイアス・微細エラー検知
- PM ループ: UX + KPI + MVQ 判定

### MVQ

単一スコアではなく specialization 軸の閾値で定義。

## 2026 の収束パターン

- **RLTHF**: LLM で easy case 自動ラベル、人間は hard case のみ → 6% の人手で full HITL 同等
- **Active Learning + Uncertainty Sampling** が RLHF パイプライン標準
- **Constitutional AI ハイブリッド** が RLAIF の bias を吸収
- **Label Studio** が OSS の最広範カバー、3 点セット (prompt/dataset/eval) は DVC + MLflow + LangSmith の組合せ運用

## このリポの実装評価

| 観点 | 判定 | 根拠 |
|---|---|---|
| 1. データ収集 | 🟡 | `mcp/ai_knowledge/tracing.py` の JSONL + Langfuse dual-write あり、agent/ts 未統合 |
| 2. ラベリング/アノテーション | ❌ | 「良い/悪い」を記録する層なし |
| 3. HITL アノテーター UI | ❌ | agent CLI に修正フィードバック機構なし |
| 4. HITL 評価ループ | ❌ | golden dataset vs 現出力の比較なし |
| 5. HITL KPI ダッシュボード | ❌ | 利用状況・コスト・品質の集計可視化なし |
| 6. データ version 管理 | ❌ | prompt / dataset / eval の 3 点セット管理なし |

## Top 2 ギャップ

### Gap 1: Feedback & Annotation Layer (10h)

```python
# mcp/agent/feedback_server.py (新規)
@mcp.tool()
def record_feedback(
    trace_id: str,
    original_output: str,
    corrected_output: str,
    is_good: bool,
    feedback_type: str  # "correctness" | "safety" | "relevance"
) -> str:
    """エージェント出力の正否・修正を記録。
    SQLite: traces × feedbacks の 1:N 関係"""
```

### Gap 2: Dashboard & Eval Loop (18h)

- FastAPI endpoint + React dashboard
- 集計: cache_hit_rate / safety_block_rate / feedback_acceptance / cost per token
- scheduled eval: golden dataset vs current model の定期比較

### Integration (8h)

- agent → feedback MCP delegate
- prompt version control (YAML + git tag)

## 累積工数

- Gap 1 (Feedback layer): 10h
- Gap 2 (Dashboard + eval loop): 18h
- Integration: 8h
- **合計: 36h (2 週間スプリント)**

## 結論と実装ロードマップ

1. **Phase 0 (10h)**: mcp/agent/feedback_server.py 新規追加、SQLite で feedback 蓄積
2. **Phase 1 (18h)**: FastAPI + React で 4 メトリクスダッシュボード、scheduled eval
3. **Phase 2 (8h)**: agent/ts → feedback MCP 統合、prompt version control

**36h で「データ収集 → ラベリング → 評価ループ → ダッシュボード」のフルサイクルが揃う**。

## 累積 8 docs 総合

- four-tradeoffs-gap-analysis.md: 26h
- architecture-and-eval-gap.md: 112h
- llmops-gap-analysis.md: 9h
- hitl-rollout-kpi-gap.md: 65h
- ux-modality-gap.md: 22h
- gui-voice-video-gap.md: 44h
- cross-modality-gap.md: 28h
- **lifecycle-hitl-gap.md（本ファイル）: 36h**
- **累積合計: 342h**

## 参考

- [The Ultimate AI Data Labeling Industry Overview 2026 (HeroHunt)](https://www.herohunt.ai/blog/the-ultimate-ai-data-labeling-industry-overview/)
- [Active Learning and HITL for LLMs (IntuitionLabs)](https://intuitionlabs.ai/articles/active-learning-hitl-llms)
- [Best Data Version Control Tools 2026 (lakeFS)](https://lakefs.io/data-version-control/dvc-tools/)
- [EU AI Act 2026 Compliance (Secure Privacy)](https://secureprivacy.ai/blog/eu-ai-act-2026-compliance)
