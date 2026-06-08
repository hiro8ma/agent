"""mini-GPT モデル (Decoder-only Transformer) の PyTorch 実装

agent/ts/transformer/ で 0 から手書きしたものを PyTorch に移植したもの。
- 構造は完全に同じ
- 違いは「重みが学習可能 (nn.Parameter) になる」「autograd で勾配が自動計算される」
- causal mask で「未来 token を見ない」を強制 → Decoder-only / GPT 系の本体構造

対応関係:
  TS 実装                              PyTorch 実装
  ─────────────                       ─────────────
  08_embedding.ts                     nn.Embedding
  09_positional_encoding.ts           nn.Embedding (学習可能 PE)
  07_multi_head_attention.ts          MultiHeadAttention class
  14_causal_mask.ts                   causal_mask buffer
  10_layer_norm.ts                    nn.LayerNorm
  11_feed_forward.ts                  FeedForward class
  12_transformer_block.ts             Block class
  13_encoder.ts (に causal mask)      MiniGPT class
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    """モデルのハイパーパラメータ"""

    vocab_size: int  # 語彙数（CharTokenizer から取る、tinyshakespeare で約 65）
    block_size: int = 128  # context 長（一度に見れる token 数）
    n_layer: int = 4  # Transformer Block の数（深さ）
    n_head: int = 4  # Multi-Head Attention のヘッド数
    n_embd: int = 128  # d_model（埋め込み次元）
    dropout: float = 0.1  # 過学習防止

    @property
    def head_dim(self) -> int:
        """1 ヘッドあたりの次元 = n_embd / n_head"""
        assert self.n_embd % self.n_head == 0, "n_embd は n_head の倍数である必要がある"
        return self.n_embd // self.n_head


class CausalSelfAttention(nn.Module):
    """Causal Mask 付き Multi-Head Self-Attention

    TS の 07_multi_head_attention.ts + 14_causal_mask.ts を統合した版。
    Q/K/V を 1 つの Linear で同時に作る最適化込み（同じ計算量だが実装が簡潔）。
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_embd = config.n_embd

        # Q/K/V を 1 つの Linear でまとめて作る（出力は (n_embd × 3)）
        # → 同じ計算量だが実装が綺麗
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)

        # Multi-Head の最終射影 (TS の W_O)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # causal mask (block_size × block_size、下三角)
        # register_buffer で「学習対象ではないが state に含める」にする
        # TS の causalMask(n) と同じ形
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, n_embd)
        B, T, C = x.shape

        # Q, K, V を一括で計算 → 3 つに分割
        qkv = self.qkv(x)  # (B, T, 3*C)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Multi-Head に分解: (B, T, C) → (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention: softmax(QK^T / √d_k) V
        # TS の 04 + 05 + 14 の統合
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)  # (B, n_head, T, T)

        # Causal mask 適用: 下三角以外を -inf にする
        scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)  # (B, n_head, T, T)
        attn = self.attn_dropout(attn)

        out = attn @ v  # (B, n_head, T, head_dim)

        # Multi-Head を結合: (B, n_head, T, head_dim) → (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # 最終射影 + dropout
        out = self.resid_dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """FFN (Feed-Forward Network)

    TS の 11_feed_forward.ts と同じ。
    数式: FFN(x) = ReLU(x W_1 + b_1) W_2 + b_2
    d_ff は 4 × n_embd（典型的な比率）
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        d_ff = 4 * config.n_embd  # 典型的な比率
        self.fc1 = nn.Linear(config.n_embd, d_ff)
        self.fc2 = nn.Linear(d_ff, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer Decoder Block（GPT 系）

    TS の 12_transformer_block.ts に causal mask を加えた版。
    Pre-Norm（LayerNorm を sublayer の前に置く）を採用 = 学習が安定する現代的な構成。
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-Norm: y = x + sublayer(LayerNorm(x))
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    """完全な mini-GPT モデル

    TS の 13_encoder.ts に causal mask を加えた = Decoder-only。
    GPT-2 などと同じ構造の最小版。
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        # Token Embedding（TS の 08）
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)

        # Positional Embedding（TS の 09 だが、固定 sin/cos の代わりに学習可能 PE）
        # 現代の GPT 系は学習可能 PE を使うことが多い
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)

        self.dropout = nn.Dropout(config.dropout)

        # Transformer Block × N
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        # 最終 LayerNorm（GPT-2 風）
        self.ln_f = nn.LayerNorm(config.n_embd)

        # 出力層: n_embd → vocab_size（次トークンのロジット）
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 重み初期化
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Xavier 風の初期化（GPT-2 と同じ標準偏差）"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """順伝播

        Args:
            idx:     (B, T) 形の token ID 列
            targets: (B, T) 形の正解 token ID 列（学習時）、None なら推論

        Returns:
            logits: (B, T, vocab_size) 各位置で「次トークン」のロジット
            loss:   targets があれば cross-entropy loss、なければ None
        """
        B, T = idx.shape
        assert T <= self.config.block_size, f"系列長 {T} が block_size {self.config.block_size} を超えた"

        # Token + Position embedding
        tok_emb = self.token_embedding(idx)  # (B, T, n_embd)
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding(pos)  # (T, n_embd)
        x = self.dropout(tok_emb + pos_emb)  # broadcast で (B, T, n_embd)

        # Transformer Block × N
        for block in self.blocks:
            x = block(x)

        # 最終 LayerNorm + 出力射影
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # 学習時のみ loss を計算
        loss = None
        if targets is not None:
            # cross_entropy は (N, vocab_size) と (N,) を期待 → reshape
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """自己回帰生成 (greedy or sampling)

        Args:
            idx:                (B, T) 開始 token 列（プロンプト）
            max_new_tokens:     何トークン生成するか
            temperature:        1.0=通常、低い→確実、高い→多様、0.0 は greedy 相当
            top_k:              上位 k 個から sampling（None なら全候補）
            frequency_penalty:  既出トークンを出現回数に比例して減算（連呼を抑制）
            presence_penalty:   既出トークンを一律減算（再登場自体を抑制）
            repetition_penalty: CTRL 流の除算ペナルティ（1.0 で無効）

        ペナルティ系はデフォルト無効なので既存の呼び出しと完全に後方互換。
        全ペナルティはプロンプト部分を除いた「生成済みトークン」のみを対象にする。

        Returns:
            (B, T + max_new_tokens) 生成された token 列
        """
        prompt_len = idx.size(1)
        for _ in range(max_new_tokens):
            # block_size を超えないように context をトリム
            idx_cond = idx[:, -self.config.block_size :]

            # forward → 最後の位置のロジットだけ取り出す
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)  # (B, vocab_size)

            # 反復ペナルティ（生成済みトークンが対象、batch ごとに適用）
            if frequency_penalty != 0.0 or presence_penalty != 0.0 or repetition_penalty != 1.0:
                logits = self._apply_repetition_penalties(
                    logits,
                    idx[:, prompt_len:],
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    repetition_penalty=repetition_penalty,
                )

            # top-k フィルタ
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # softmax → サンプリング
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # 生成結果を追記
            idx = torch.cat([idx, idx_next], dim=1)

        return idx

    @staticmethod
    def _apply_repetition_penalties(
        logits: torch.Tensor,
        generated: torch.Tensor,
        *,
        frequency_penalty: float,
        presence_penalty: float,
        repetition_penalty: float,
    ) -> torch.Tensor:
        """既出トークンのロジットを抑制した新しいロジットを返す（バッチ対応）。

        logits:    (B, vocab) 各 batch の次トークンロジット
        generated: (B, G)     これまでに生成したトークン ID（プロンプトは除く）
        """
        if generated.size(1) == 0:
            return logits
        out = logits.clone()
        vocab = out.size(-1)
        for b in range(out.size(0)):
            counts = torch.bincount(generated[b], minlength=vocab).to(out.dtype)
            seen = counts > 0
            if frequency_penalty != 0.0:
                out[b] = out[b] - frequency_penalty * counts
            if presence_penalty != 0.0:
                out[b] = out[b] - presence_penalty * seen.to(out.dtype)
            if repetition_penalty != 1.0:
                pos = out[b] > 0
                lowered = out[b] / repetition_penalty
                raised = out[b] * repetition_penalty
                out[b] = torch.where(seen, torch.where(pos, lowered, raised), out[b])
        return out

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# === 動作確認 ===
# 直接実行: uv run python model.py
if __name__ == "__main__":
    print("=== mini-GPT モデル構造の動作確認 ===\n")

    config = GPTConfig(
        vocab_size=65,  # tinyshakespeare の典型値
        block_size=32,
        n_layer=2,
        n_head=2,
        n_embd=64,
        dropout=0.0,
    )

    model = MiniGPT(config)
    print(f"パラメータ数: {model.num_parameters():,}")
    print(f"\nモデル構造:")
    print(model)

    print("\n=== Forward の動作確認 ===")
    B, T = 4, 8
    idx = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))

    logits, loss = model(idx, targets)
    print(f"入力 idx.shape    = {tuple(idx.shape)}  (B={B}, T={T})")
    print(f"出力 logits.shape = {tuple(logits.shape)}  (B, T, vocab_size)")
    print(f"loss              = {loss.item():.4f}")
    print(f"  (random init では log(vocab_size)={torch.log(torch.tensor(config.vocab_size)).item():.4f} 程度のはず)")

    print("\n=== 推論時 (loss なし) ===")
    logits, loss = model(idx)
    print(f"loss = {loss}  (None になるのが期待動作)")

    print("\n=== 生成の動作確認（学習前なのでランダムな文字列が出る）===")
    start = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(start, max_new_tokens=20)
    print(f"生成された token IDs: {out[0].tolist()}")
    print(f"形: {tuple(out.shape)} = (B=1, T=1+20)")
