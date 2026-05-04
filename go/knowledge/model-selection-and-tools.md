---
title: "モデル選定とツール設計の戦略"
date: "2026-05-04"
tags: [agent, model-selection, tool-design, hybrid-routing]
---

# モデル選定とツール設計の戦略

エージェントの価値は「モデル × ツール × 思考ループ」の積で決まる。モデル選定はトレードオフ設計、Tool は agent の手足。

## モデル選定 — 5 つの軸

| 軸 | 観点 |
|---|---|
| タスクの複雑さ | 推論深さ / 曖昧さ / 創造性が必要か |
| レイテンシー | リアルタイム vs バッチ、UX への影響 |
| コスト | per token 単価 × 想定トラフィック |
| 精度 / 柔軟性 | 構造化出力の安定性、エッジケース対応 |
| インフラ制約 | クラウド / オンプレ / エッジ、規制要件 |

これらは互いにトレードオフ関係。1 軸の最適化が他軸を犠牲にする。

## 大規模モデル vs 小規模モデル

| | 大規模モデル | 小規模モデル |
|---|---|---|
| 例（2026） | GPT-5.x / Claude Opus 4.7 / Gemini 2.5 Pro | Phi-4 / Llama 3.2 / Gemma 3 / Mistral Small |
| 強み | 推論・曖昧さ・創造性 | 高速・安価・ローカル可 |
| 弱み | コスト高・遅い | 柔軟性低い |
| 向き | オープンエンド / 不確実なタスク | 定型処理 / 構造化タスク |

## モダリティ

入力 / 出力に何を扱うかでモデル選定が変わる。

- テキストのみ: ほぼ全モデル対応
- 画像入力: GPT-4o / Claude / Gemini / Llava 系
- 音声入出力: GPT Realtime / Gemini Live API
- 構造データ: code-tuned モデル（Claude Sonnet / DeepSeek-Coder 等）

## オープン vs プロプライエタリ

| | オープン | プロプライエタリ API |
|---|---|---|
| 例 | Llama 3.x / Gemma 3 / Mistral / DeepSeek | GPT / Claude / Gemini |
| 強み | 柔軟、カスタム可、自前運用、データ秘匿 | すぐ使える、高性能、運用負荷低 |
| 弱み | 運用コスト増、品質ばらつき | コスト増、ベンダーロック |
| 向き | データ機密性 / 大量推論 / fine-tuning 必須 | プロトタイプ / 高品質要求 |

## ハイブリッド戦略

```
入力 → ルーター（小モデル or ルールベース）
         ├── 簡単な処理 → 小規模モデル
         ├── 中程度の処理 → 中規模モデル
         └── 難しい処理 → 大規模モデル
```

実装パターン:

- **ルーター**: 小モデル / 分類器 / ルールで分岐判断
- **キャッシュ**: 同一クエリは embedding 比較で短絡
- **エスカレーション**: 小モデルの confidence が低ければ大モデルへ
- **Fallback**: メインモデル失敗時に別モデルへ

実装ツール:

- Portkey / OpenRouter / LiteLLM: マルチプロバイダ + ルーティング
- AWS Bedrock Intelligent Prompt Routing
- Vertex AI Model Garden

## ベンチマークの扱い

MMLU / HellaSwag / HumanEval などの公開ベンチマークは **参考程度**。

理由:

- データ汚染（モデルがベンチマークデータを学習）
- 自分のユースケースとの乖離
- 実運用での失敗パターンは公開ベンチに出ない

→ **「自分のタスクで評価」が最重要**。Golden QA / Ragas / LLM-as-judge を自分のユースケースで構築する。

## モデル選定は継続的な意思決定

新モデルが出るたびに評価 / 入れ替え検討、A/B テストでのプロモーション、部分的入れ替え（特定 Skill だけモデル変更）。

→ Provider 抽象化レイヤーが活きる。

## ツール設計 — エージェントの手足

エージェントの価値は「モデル」だけでなく「Tool 設計」で決まる。

### ツールの 3 役割

1. **外部 API 呼び出し**: SaaS / 内部システム / 公開 API
2. **データ取得**: DB / file / search engine
3. **アクション実行**: 状態変更 / 通知 / PR 作成

### Tool 設計の原則

- **粒度**: coarse-grained vs fine-grained — 後者が LLM のディスパッチ精度を上げる
- **idempotency**: 同じ呼び出しが同じ結果を返す
- **dry-run**: 破壊的操作には事前確認モード
- **needsApproval**: HITL 承認フックを Tool 単位で持つ
- **allowlist**: 公開する Tool を annotation で明示（デフォルト deny）

## まとめ

> モデル選定は「精度・コスト・速度の最適化問題」、実運用ではハイブリッド構成 + Tool 設計が勝負。

## 関連

- `agent-system-design-principles.md` — 5 原則
- `agent-frameworks-comparison-2026.md` — フレームワーク選択
- `agent-scope-setting.md` — スコープ設計
- `coding-agent-design-patterns.md` — 実装パターン
