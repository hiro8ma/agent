---
title: "プロンプトエンジニアリング — ICL / Few-Shot / CoT（2026）"
date: "2026-05-05"
tags: [prompt-engineering, few-shot, cot, icl, evaluation]
---

# プロンプトエンジニアリング — ICL / Few-Shot / CoT

LLM の出力品質を上げる基本テクニック。In-Context Learning（ICL）の派生として Few-Shot / Chain of Thought（CoT）がある。

## In-Context Learning（ICL, 文脈内学習）

LLM は **プロンプトから新しいタスクを学習し、その学習情報を使って新たな出力を生成する** 能力を持つ。fine-tuning なしで、プロンプト内に「タスクの例」を埋め込むだけで望ましい挙動を引き出せる。

## Few-Shot Prompting

| 名称 | 例の数 |
|---|---|
| Zero-Shot | 0 個（タスク説明のみ） |
| One-Shot | 1 個 |
| Few-Shot | 複数 |

### 例: 二値分類

```
以下の Text から 0 か 1 かを判定してください。回答のみを答えてください。
Text: これはすばらしい！
Answer: 1
Text: これは悪いことだ！
Answer: 0
Text: あの映画はすばらしかった！
Answer: 1
Text: なんてひどいショーなんだ！
Answer:
```

0 / 1 のラベル意味は明示していないが、プロンプトから対比を学習して正しく判定する。**ラベルの意味よりも、ラベル間の対比を学んでいる**。

### Few-Shot が効きやすいタスク

- 二値 / 多値分類
- フォーマット変換（JSON / SQL / Markdown 等）
- スタイル統一（出力構造の安定化）
- intent 検出 / ルーティング

### 効果的な Few-Shot の作り方

- **例は 3-5 個**（多すぎると context 圧迫、少なすぎると揺らぐ）
- **多様性を持たせる**（境界事例を含める）
- **エッジケースを明示**（曖昧な入力の処理を例で示す）
- **同じフォーマット**（区切り文字 / 改行を統一）
- **最後に問題本体を置く**（recency bias を活用）

## Chain of Thought（CoT）

「答えを出す前に推論過程を書いて」と指示することで、複雑な推論タスクの精度を上げる手法。

### Few-Shot CoT 例

```
問題: A から B まで 10km を時速 5km で歩く。何時間かかる？
推論: 距離 ÷ 速度 = 10 ÷ 5 = 2
答え: 2 時間

問題: B から C まで 30km を時速 60km の車で行く。何分かかる？
推論: 30 ÷ 60 = 0.5 時間 = 30 分
答え: 30 分

問題: A から D まで 100km、最初の 50km は時速 50km、残りは時速 100km。合計時間は？
推論:
答え:
```

### 2026 年時点の CoT の位置付け

- **大規模モデル（GPT-5.x / Claude Opus 4.7 / o1 / o3）**: モデル側に reasoning 機構が内蔵、明示 CoT の価値は下がっている
- **`<thinking>` タグ系**: Claude の extended thinking は明示 CoT の進化形
- **小規模モデル（Phi / Llama / Gemma 等）**: 明示 CoT を入れると精度大幅 UP、依然として有効
- **Tool 選択の判断可視化**: 中規模モデルでも `<thinking>...</thinking>` を返すよう促すと debug しやすい

### ハイブリッド戦略との接続

- 小モデルに CoT で `confidence` を出させて router の判断材料にする
- 「自信がない」と言ってきたら大モデルにエスカレーション

## 日本語評価ベンチマーク

LLM のモデル選定 / 改修時に使える公開ベンチマーク。

| ベンチマーク | 特徴 |
|---|---|
| ELyZA-tasks-100 | 100 件の多様な複雑タスク、人間 / LLM-as-judge どちらでも評価可 |
| Nejumi | 日本語 LLM 総合評価リーダーボード（W&B 主導） |
| Japanese MT-Bench | 多ターン対話評価 |
| JCommonSenseQA | 日本語常識推論 |
| JGLUE | 日本語版 GLUE |

### 評価フロー（LLM-as-judge）

1. 評価対象モデルにベンチマークを解かせる
2. 出力を判定モデル（GPT-5 等）に渡して採点
3. 平均スコアでモデル比較

### 公開ベンチマーク vs 自社 Golden QA

| | 公開ベンチ | 自社 Golden QA |
|---|---|---|
| 用途 | 汎用性 sanity check | ドメイン特化品質保証 |
| 利点 | 独立性、業界標準 | 自社ユースケースに直結 |
| 欠点 | 自社タスクと乖離 | 作成 / 維持コスト |

両者を組み合わせるのが定石。

## 実装上の注意点

```python
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
response = client.responses.create(
    model=os.environ["LLM_MODEL"],   # ハードコードしない
    input=PROMPT,
    temperature=0,                    # 評価は再現性のため 0
    max_output_tokens=500,            # コスト暴走防止
)
```

- **モデル名のハードコード禁止**: 半年で陳腐化、env or config で外出し
- **Responses API**: `chat.completions` は legacy、新規は `responses.create`
- **評価用途は `temperature=0`**: 実験再現性のため
- **`max_tokens` の上限設定**: コスト暴走防止
- **エラーハンドリング**: rate limit / safety filter / timeout を分類してリトライ可否判断

## 5 原則との接続

`agent-system-design-principles.md` の 5 原則と接続:

- **継続的学習**: 公開ベンチ + Golden QA で評価ループ
- **モジュール性**: Few-Shot 例を外部化、prompt から分離
- **将来性**: モデル切替時に同じベンチで sanity check

## 参考

- ELyZA-tasks-100: https://huggingface.co/datasets/elyza/ELYZA-tasks-100
- Nejumi: https://wandb.ai/llm-jp-eval/nejumi-leaderboard
- "Chain-of-Thought Prompting Elicits Reasoning" (Wei et al., 2022): https://arxiv.org/abs/2201.11903
- "Language Models are Few-Shot Learners" (Brown et al., 2020): https://arxiv.org/abs/2005.14165

## 関連

- `agent-system-design-principles.md` — 5 原則
- `model-selection-and-tools.md` — モデル選定とツール設計
- `agent-scope-setting.md` — スコープ設計
