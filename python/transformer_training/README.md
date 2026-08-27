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
| `word_bpe_demo.py` | 教材向けの単語頻度つき文字ベース BPE。`split()` → word frequency → char vocab → pair frequency → merge loop を最小コードで可視化 |
| `bpe.py` | バイト単位 tokenizer + BPE（train / encode / decode / save / load）+ 事前トークン化（GPT-2 正規表現で単語 / 数字 / 句読点 / 短縮形 / 空白へ分割し、マージを pretoken 内部に閉じ込める。`regex` 依存）+ 特殊トークン `<\|endoftext\|>`（学習は区切りで分割、encode は 1 ID に圧縮、`allow_special=False` で解釈無効化）。char-level の代替として差し替え可能 |
| `eval_tokenizer.py` | tokenizer の圧縮率評価。Char vs BPE の比較 + 語彙サイズ別（300/500/1000）の圧縮率と埋め込み params のトレードオフを表で観察 |
| `encode_to_bin.py` | コーパスを BPE でエンコードし `uint16` バイナリ（`train.bin` / `val.bin`）へ永続化。読み戻しで要素数一致と decode 往復を検証（nanoGPT 流） |
| `model.py` | Decoder-only Transformer (mini-GPT) の PyTorch 実装 |
| `train.py` | 学習ループ（forward, loss, backward, optimizer.step）。デフォルトは char-level、`--tokenizer bpe` で BPE/bin pipeline に切替 |
| `generate.py` | 学習済みモデルで文章生成。checkpoint 内の tokenizer 種別（char / BPE）を復元 |

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

# 3b. BPE/bin pipeline で学習（初回は tokenizer と .bin を自動生成）
make train-bpe

# 4. 学習済みモデルで生成
make generate

# 5. プロンプト指定して生成
make generate-romeo
```

## Tokenizer の圧縮率評価と bin 化

```bash
# 圧縮率を評価（Char vs BPE、語彙サイズ別トレードオフ）
make eval-tokenizer

# 単語頻度つき文字ベース BPE の教育用デモ
make word-bpe

# コーパスを uint16 .bin に永続化（train.bin / val.bin + 読み戻し検証）
make encode-bin
```

### word-level BPE デモ

`word_bpe_demo.py` は GPT 系の byte-level BPE ではなく、教材でよく出る「単語頻度を
重みづけした文字ペア BPE」を見せるためのデモ。`processing.` のように句読点が単語に
付いたまま残る素朴な `split()` から始め、頻出隣接ペアを 10 回 merge する。

このデモの役割は、`bpe.py` の実用寄り byte-level BPE を理解する前段。

### 圧縮率（compression ratio）

`byte 数 / token 数`（1 トークンあたりの平均バイト数）。値が大きいほど列が短く、同じ context 長でより多くの情報を扱える。tinyshakespeare 先頭 50,000 文字での比較

| tokenizer | vocab | token 数 | 圧縮率 |
|---|---|---|---|
| Char | 65 | 50,000 | 1.00 倍 |
| BPE | 300 | 35,696 | 1.40 倍 |
| BPE | 500 | 24,899 | 2.01 倍 |
| BPE | 1000 | 19,000 | 2.63 倍 |

語彙を増やすと圧縮率は上がるが、埋め込み層（`vocab_size * n_embd`）も線形に増える。圧縮率と埋め込みコストのバランスで実用 vocab（GPT-2 は 50,257）が決まる。

### uint16 .bin 永続化

`encode_to_bin.py` はコーパスを BPE でエンコードし、先頭 90% を `data/train.bin`、残りを `data/val.bin` に書き出す（時系列順）。token ID は `np.uint16` で持つ（`int32` 比でディスク/メモリ半減、vocab <= 65535 が前提。超える場合は `uint32` にフォールバック）。学習ループは `np.fromfile` で即ロードでき、毎回のエンコードを省ける。

`.bin` と tokenizer の merges（`data/bpe_tinyshakespeare.json`）は再生成可能なので git 管理しない（`.gitignore` 済）。

`train.py --tokenizer bpe` は target vocab ごとに以下を自動生成してから学習する。

```text
data/bpe_tinyshakespeare_vocab300.json
data/train_bpe_tinyshakespeare_vocab300.bin
data/val_bpe_tinyshakespeare_vocab300.bin
```

checkpoint には tokenizer 種別と BPE merge ルールを保存するため、`generate.py` は char-level / BPE のどちらで学習した checkpoint でも同じコマンドで復元できる。

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

## データ並列（DDP）

`ddp_demo.py` — 勾配平均が本当に起きているかを対照実験で確かめる
`ddp_train.py` — mini-GPT の学習を DDP で並列化する

```bash
uv run python ddp_demo.py --world-size 4
uv run python ddp_train.py --world-size 2 --max-iters 200
```

CUDA が無くても動く。バックエンドは `gloo` で、CPU プロセスを並べる。
教材で指定される `nccl` は GPU 間通信のライブラリで CUDA が要るため、
Apple Silicon では選べない。

### なぜ対照実験が要るか

DDP を付けても付けなくても学習は進む。損失は下がるし、エラーも出ない。
「動いた」ことは勾配が平均されている証明にならない。

`ddp_demo.py` は同じ初期値・別々のデータを各ランクに与えて逆伝播し、
ランク間で勾配が一致するかを見る。実測結果は次のとおり。

```
DDP なし  rank0 [-1.846495, 1.264056, -2.260092, -2.088561]
          rank1 [-0.605713, 0.550910, -0.603843, -0.012567]   食い違う
DDP あり  rank0 [-0.888836, 1.050793, -0.976977, -0.618419]
          rank1 [-0.888836, 1.050793, -0.976977, -0.618419]   一致する

DDP なしの平均 = [-0.888836, 1.050793, -0.976977, -0.618419]  ← 完全一致
```

### DistributedSampler を使わない理由

この学習は DataLoader ではなく、データ列からランダムなオフセットを切り出す方式になる。
ランクごとに乱数の種を変えれば重複しないバッチが得られるため、Sampler は要らない。

Sampler はインデックス列を分割する仕組みで、この方式には噛み合わない。
教材が使っているからという理由で入れると、機構が合わないまま動いてしまう。

一方 DataLoader を使う場合は Sampler が要る。外すと全ランクが同じデータを見る。
`ddp_demo.py` の最後でその状態を再現している。エラーは出ず、
実質バッチサイズが world_size 倍になるだけで学習は進む。

## テンソル並列と ZeRO

`tensor_parallel_demo.py` — 分解が数学的に厳密であることと通信量の差
`zero_fsdp_demo.py` — ZeRO（FSDP）がパラメータを分散保持していることの実測

```bash
uv run python tensor_parallel_demo.py
uv run python zero_fsdp_demo.py --world-size 2
```

### なぜ 1 層目を列、2 層目を行で割るのか

方向は選べる自由度ではなく、モデルの構造が決める。

```
Colwise  出力側の次元を分ける。入力は全次元が要る
Rowwise  入力側の次元を分ける。出力は部分和になる
```

1 層目を Colwise にすると隠れ層が列方向に分かれる。
それを受ける 2 層目は、入力側が分かれている前提の Rowwise でなければ形が合わない。

両方を Colwise にすると、層をまたぐたびに all-gather で隠れ層を集める必要が出る。
Colwise → Rowwise なら最後に all-reduce が 1 回で済む。

```
all-gather で動く要素数  = 隠れ層の全体
all-reduce で動く要素数  = 出力のみ
隠れ層が出力より広いほど差が開く
```

分解は近似ではない。実測した最大誤差は 2.38e-07 で、浮動小数の丸めだけになる。
2 分割でも 4 分割でも単一 GPU の結果と一致する。

### ZeRO は「動いた」では確かめられない

DDP と FSDP は、どちらも loss が同じように下がる。
違いは各ランクが手元に持つパラメータ量にしかない。

```
DDP   rank0 263,425 要素（100%）  rank1 263,425 要素（100%）  合計 = 全体 × 2.0
FSDP  rank0 131,841 要素（ 50%）  rank1 131,584 要素（ 50%）  合計 = 全体 × 1.0
```

loss は両方とも 1.2512 -> 1.2065 で完全に同じだった。
分散保持されているかは、保持している要素数を数える以外に判別方法がない。

### Apple Silicon での注意

`fully_shard` に `mesh` を渡さないと既定デバイスの判定が走り、
MPS と見なされて `torch.mps.is_initialized` が無いために落ちる。
`init_device_mesh("cpu", (world_size,))` を明示して渡す。
