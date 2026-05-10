# transformer_training

Transformer を **0 から学習する** 教育用 mini-GPT 実装（PyTorch）。

`agent/ts/transformer/` で 0 から手書きしたアーキテクチャを **PyTorch に移植 + 学習を実装** したもの。Andrej Karpathy "Let's build GPT" シリーズと同じ構成。

## 目的

`ts/transformer/` まででは「アーキテクチャ（計算の設計図）」を完成させた。
このリポジトリは「**学習アルゴリズム**」を実装する。

学習プロセス
1. **Forward** モデルに入力 → 各位置で次トークンの予測
2. **Loss** cross-entropy で予測 vs 正解の差を数値化
3. **Backward** autograd で全パラメータの勾配を計算
4. **Update** AdamW で重みを少し動かす

これを数千ステップ繰り返すと、ランダム初期化された重みが「シェイクスピアっぽい英文を生成する重み」に変わる。

## 構成

| ファイル | 役割 |
|---|---|
| `data.py` | tinyshakespeare ダウンロード + char-level tokenizer + DataLoader |
| `model.py` | Decoder-only Transformer (mini-GPT) の PyTorch 実装 |
| `train.py` | 学習ループ（forward, loss, backward, optimizer.step） |
| `generate.py` | 学習済みモデルで文章生成 |

## TS 実装との対応

| TS (`agent/ts/transformer/`) | PyTorch (`model.py`) |
|---|---|
| 08_embedding.ts | `nn.Embedding` |
| 09_positional_encoding.ts | `nn.Embedding`（学習可能 PE） |
| 07_multi_head_attention.ts + 14_causal_mask.ts | `CausalSelfAttention` |
| 11_feed_forward.ts | `FeedForward` |
| 10_layer_norm.ts | `nn.LayerNorm` |
| 12_transformer_block.ts | `Block` |
| 13_encoder.ts (+ causal mask) | `MiniGPT` |

**構造は完全に同じ**。違いは「重みが学習可能 (`nn.Parameter`)」「autograd で勾配が自動計算される」の 2 点。

## セットアップ

```bash
cd /Users/hiroma/go/src/github.com/hiro8ma/agent/python/transformer_training
make setup
```

`uv sync` で torch などの依存関係をインストール。

## 動かし方

```bash
# 1. データセットの動作確認（tinyshakespeare ダウンロード + tokenizer 確認）
make data

# 2. モデル構造の動作確認（学習前）
make model

# 3. 学習開始（M2 Max GPU で約 5-10 分、CPU でも 20-30 分）
make train

# 4. 学習済みモデルで生成
make generate

# 5. プロンプト指定して生成
make generate-romeo
```

## 学習の進捗の見方

```
step    0 | train loss 4.17 | val loss 4.17  ← ランダム初期化
step  200 | train loss 2.50 | val loss 2.50  ← 文字頻度を学んだ
step 1000 | train loss 1.80 | val loss 1.85  ← 単語っぽい構造を学んだ
step 3000 | train loss 1.50 | val loss 1.60  ← 英文っぽくなる
```

`log(vocab_size) = log(65) ≈ 4.17` から始まり、下がっていく。
**val loss が train loss から離れ始めたら過学習開始のサイン**。

## 学習途中での生成サンプル

500 ステップごとに生成サンプルが出力される。学習の進化が観察できる

```
step 500   生成: ".d  .e: t.   so .   o no ot uvenec o  s..." （ノイズ）
step 1500  生成: "And shall I king the truth and..."           （単語っぽい）
step 3000  生成: "ROMEO: My lord, I have not seen this..."    （文っぽい）
```

## ハイパーパラメータ

`train.py` の `TrainConfig` で調整可能

| パラメータ | デフォルト | 意味 |
|---|---|---|
| `block_size` | 128 | context 長 |
| `batch_size` | 32 | 1 ステップで処理するサンプル数 |
| `n_layer` | 4 | Transformer Block の数 |
| `n_head` | 4 | Multi-Head Attention のヘッド数 |
| `n_embd` | 128 | d_model（埋め込み次元） |
| `max_iters` | 3000 | 学習ステップ数 |
| `learning_rate` | 3e-4 | AdamW の lr (GPT-2 と同じ) |
| `weight_decay` | 0.1 | AdamW の Weight Decay |
| `grad_clip` | 1.0 | 勾配ノルムの上限 |
| `warmup_iters` | 100 | learning rate warmup ステップ数 |

## 注意

- 重みは tinyshakespeare で学習されるので、**シェイクスピア風の英文しか生成できない**
- vocab は character-level なので、学習データに無い文字（日本語など）はプロンプトに使えない
- これはあくまで教育目的、実用は Hugging Face / OpenAI / Anthropic API を使う
