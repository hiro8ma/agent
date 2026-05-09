# transformer

Scaled Dot-Product Attention を **数式 1:1 対応のコード** でスクラッチ実装する学習用パッケージ。

外部 ML ライブラリ（numpy / tfjs / pytorch 等）を使わず、`number[][]` だけで書く。1 ファイル 1 概念で積み上げる構成。各ファイルは末尾に `if (import.meta.main)` の動作確認セクションを持ち、`bun run` で個別に動かして中間結果を確認できる。

## 構成

| ファイル | 概念 | 数式 |
|---|---|---|
| `01_softmax.ts` | softmax | `softmax(a_i) = e^{a_i} / Σ e^{a_j}` |
| `02_dot_product.ts` | ベクトル内積 | `⟨q, k⟩ = Σ q_i k_i` |
| `03_qk_similarity.ts` | Query × 複数 Keys | `Q K^T : (n_q × d_k) × (d_k × n_k) → (n_q × n_k)` |
| `04_scaled_dot_product.ts` | √d_k スケーリング | `(Q K^T) / √d_k` |
| `05_attention.ts` | Attention 完全版 | `Attention(Q,K,V) = softmax(Q K^T / √d_k) V` |
| `06_linear_projection.ts` | 線形変換（W^Q / W^K / W^V） | `y = x W` |
| `07_multi_head_attention.ts` | Multi-Head Attention | `O = concat(O_1, ..., O_h) W^O`、各 `O_i = Attention(Q W^Q_i, K W^K_i, V W^V_i)` |
| `08_embedding.ts` | Embedding（埋め込み） | `embed(id) = E[id]` |
| `09_positional_encoding.ts` | Positional Encoding | `PE(i, 2k) = sin(i/10000^{2k/d})`, `PE(i, 2k+1) = cos(i/10000^{2k/d})` |
| `10_layer_norm.ts` | Residual + LayerNorm | `y = LayerNorm(x + sublayer(x))` |
| `11_feed_forward.ts` | Feed-Forward Network | `FFN(x) = ReLU(x W_1 + b_1) W_2 + b_2` |
| `12_transformer_block.ts` | Transformer Block 統合 | フルパイプライン（Embedding → PE → MHA → R+LN → FFN → R+LN）|
| `13_encoder.ts` | Transformer Encoder（N 層スタック）| 12 を N 層直列に通す。BERT 等の Encoder-only モデル本体 |

## 動かし方

```bash
cd ts

# 各 Step を個別に実行（中間結果が見える）
bun run transformer/01_softmax.ts
bun run transformer/02_dot_product.ts
bun run transformer/03_qk_similarity.ts
bun run transformer/04_scaled_dot_product.ts
bun run transformer/05_attention.ts
```

## 数式 ↔ 直感の対応

### Attention の本質

「**Query に似ている Key を探し、その Value を強く取り込む**」仕組み。

| 概念 | 役割 |
|---|---|
| Query (Q) | 何を探したいか |
| Key (K) | 何を持っているか |
| Value (V) | 実際に取り込みたい情報 |
| Q K^T | Query と各 Key の類似度（向きが近いほど大） |
| / √d_k | スコアの暴走を防ぐスケーリング |
| softmax | 重みを確率分布化（合計 1、大きい値ほど強調） |
| × V | 重みで Value を加重平均 |

### 「向きが近いほど内積が大きい」

```
q = [1, 0]    （右向き）
k1 = [1, 0]   → ⟨q, k1⟩ = 1   （同じ向き、注目）
k2 = [0, 1]   → ⟨q, k2⟩ = 0   （直交、無視）
k3 = [-1, 0]  → ⟨q, k3⟩ = -1  （反対、抑制）
```

これが「Attention は意味空間で類似度検索している」と言われる正体。

### softmax の本質

「大きい値をさらに強調する」関数。`[1, 2, 10]` → `[ほぼ0, ほぼ0, ほぼ1]`。

数値安定化のため `max` を引いてから `exp` を取る（`overflow` 防止、結果は数学的に等価）。

### √d_k で割る理由

次元 `d_k` が大きいと内積が大きくなりやすく、softmax が極端化（one-hot 寄り）→ 勾配消失。`√d_k` で割ることでスコアの分散を抑え、softmax が穏やかになる。

## 実装で省いていること（あえて）

- バッチ処理（最初の Q が 1 つの場合に限定）
- マスキング（causal mask、padding mask）
- Multi-Head Attention（複数の注意ヘッドを並列）
- Dropout
- 学習可能パラメータ（W_Q, W_K, W_V）

これらは別 commit で段階的に追加する予定。**まず内積アテンションの 4 段（生スコア → スケール → softmax → V 合成）を完璧に理解する** ことを優先。

## 参考

- mcp/transformer/notebooks/03_multi_head_attention.ipynb（numpy 版、より詳細な解説）
- ai/knowledge/transformer/dot-product-attention.md
- ai/knowledge/transformer/additive-vs-dot-product-attention.md
- ai/knowledge/transformer/softmax-numerical-stability.md
- "Attention Is All You Need" (Vaswani et al., 2017)
