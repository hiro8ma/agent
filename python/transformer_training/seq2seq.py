"""Encoder-Decoder Transformer (seq2seq) の PyTorch 実装

model.py の MiniGPT は Decoder-only（causal self-attention のみ）。これは元論文
"Attention Is All You Need" の完全な Encoder-Decoder 版。翻訳・要約・系列変換のように
「入力系列を読み切ってから別の系列を生成する」タスク向けの構造。

3 種類の attention が登場する:
  1. Encoder self-attention        双方向（mask は src の padding のみ）
  2. Decoder masked self-attention causal + tgt padding（未来 token を見ない）
  3. Decoder cross-attention       Q=デコーダ, K/V=エンコーダ出力（src padding mask）

cross-attention が Encoder と Decoder を繋ぐ橋。デコーダの各位置が「入力系列の
どこを見るべきか」をエンコーダ出力に対して引く。Decoder-only との本質的な差分はここ。

model.py との対応:
  CausalSelfAttention → MultiHeadAttention（Q と K/V を別入力で受ける汎用版に一般化）
  FeedForward         → 同じ構造を再掲（GPTConfig 非依存にするため独立定義）
  Block               → EncoderBlock / DecoderBlock に分化
  MiniGPT             → Seq2SeqTransformer（Encoder + Decoder）
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Seq2SeqConfig:
    """Encoder-Decoder モデルのハイパーパラメータ"""

    src_vocab_size: int  # 入力（ソース）語彙数
    tgt_vocab_size: int  # 出力（ターゲット）語彙数
    max_len: int = 64  # 位置エンコーディングが扱える最大系列長
    n_layer: int = 3  # Encoder / Decoder それぞれの層数
    n_head: int = 4  # Multi-Head Attention のヘッド数
    d_model: int = 128  # 埋め込み次元
    dropout: float = 0.1

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_head == 0, "d_model は n_head の倍数である必要がある"
        return self.d_model // self.n_head


class MultiHeadAttention(nn.Module):
    """汎用 Multi-Head Attention（self / cross 両対応）

    model.py の CausalSelfAttention は Q/K/V を同一入力から 1 つの Linear で作っていた。
    cross-attention では Q がデコーダ、K/V がエンコーダ出力と「出所が異なる」ため、
    Q 用と K/V 用の Linear を分けて、forward で query と key_value を別々に受ける。

    mask: (B, 1, T_q, T_k) のブール（True=見てよい / False=-inf）。causal も padding も
    呼び出し側で合成して渡す。この class はマスクの中身を解釈しない。
    """

    def __init__(self, d_model: int, n_head: int, dropout: float) -> None:
        super().__init__()
        assert d_model % n_head == 0, "d_model は n_head の倍数である必要がある"
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.d_model = d_model

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # query:     (B, T_q, d_model)  self-attn では key_value と同一テンソル
        # key_value: (B, T_k, d_model)  cross-attn ではエンコーダ出力
        B, T_q, C = query.shape
        T_k = key_value.size(1)

        q = self.q_proj(query)  # (B, T_q, C)
        kv = self.kv_proj(key_value)  # (B, T_k, 2C)
        k, v = kv.split(self.d_model, dim=2)

        # Multi-Head 分解: (B, T, C) → (B, n_head, T, head_dim)
        q = q.view(B, T_q, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T_k, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T_k, self.n_head, self.head_dim).transpose(1, 2)

        # scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, n_head, T_q, T_k)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = attn @ v  # (B, n_head, T_q, head_dim)

        out = out.transpose(1, 2).contiguous().view(B, T_q, C)
        out = self.resid_dropout(self.out_proj(out))
        return out


class FeedForward(nn.Module):
    """FFN: FFN(x) = ReLU(x W_1) W_2。d_ff は 4 × d_model（典型比率）"""

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        d_ff = 4 * d_model
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class EncoderBlock(nn.Module):
    """Encoder Block: 双方向 self-attention + FFN（Pre-Norm + 残差）

    causal mask は無い。src の padding mask のみ受ける（双方向に全 token を見れる）。
    """

    def __init__(self, config: Seq2SeqConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.self_attn = MultiHeadAttention(config.d_model, config.n_head, config.dropout)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config.d_model, config.dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor | None) -> torch.Tensor:
        h = self.ln1(x)
        x = x + self.self_attn(h, h, src_mask)
        x = x + self.ffn(self.ln2(x))
        return x


class DecoderBlock(nn.Module):
    """Decoder Block: masked self-attention + cross-attention + FFN

    3 sublayer 構成（Pre-Norm + 残差を各 sublayer に適用）:
      1. masked self-attention   tgt_mask = causal ∧ tgt padding
      2. cross-attention         Q=デコーダ状態, K/V=エンコーダ出力, mask=src padding
      3. FFN
    """

    def __init__(self, config: Seq2SeqConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.self_attn = MultiHeadAttention(config.d_model, config.n_head, config.dropout)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.cross_attn = MultiHeadAttention(config.d_model, config.n_head, config.dropout)
        self.ln3 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config.d_model, config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None,
        src_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        h = self.ln1(x)
        x = x + self.self_attn(h, h, tgt_mask)
        # cross-attention: Q はデコーダ状態、K/V はエンコーダ出力 memory（橋渡し）
        h = self.ln2(x)
        x = x + self.cross_attn(h, memory, src_mask)
        x = x + self.ffn(self.ln3(x))
        return x


def _positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """固定 sin/cos 位置エンコーディング (max_len, d_model)（元論文方式）"""
    pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class Encoder(nn.Module):
    """token embedding + 位置エンコーディング + EncoderBlock × N"""

    def __init__(self, config: Seq2SeqConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.token_embedding = nn.Embedding(config.src_vocab_size, config.d_model)
        self.register_buffer("pos_enc", _positional_encoding(config.max_len, config.d_model))
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([EncoderBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.d_model)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor | None) -> torch.Tensor:
        T = src.size(1)
        x = self.token_embedding(src) * math.sqrt(self.d_model)
        x = self.dropout(x + self.pos_enc[:T])
        for block in self.blocks:
            x = block(x, src_mask)
        return self.ln_f(x)


class Decoder(nn.Module):
    """token embedding + 位置エンコーディング + DecoderBlock × N + 出力射影"""

    def __init__(self, config: Seq2SeqConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.token_embedding = nn.Embedding(config.tgt_vocab_size, config.d_model)
        self.register_buffer("pos_enc", _positional_encoding(config.max_len, config.d_model))
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.tgt_vocab_size, bias=False)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None,
        src_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        T = tgt.size(1)
        x = self.token_embedding(tgt) * math.sqrt(self.d_model)
        x = self.dropout(x + self.pos_enc[:T])
        for block in self.blocks:
            x = block(x, memory, tgt_mask, src_mask)
        x = self.ln_f(x)
        return self.lm_head(x)  # (B, T_tgt, tgt_vocab_size)


def create_padding_mask(seq: torch.Tensor, pad_id: int) -> torch.Tensor:
    """padding mask (B, 1, 1, T) を作る。True=実トークン / False=pad（-inf 対象）。

    key 軸（最後の次元）に対するマスク。pad 位置を attention の参照先から除外する。
    """
    return (seq != pad_id).unsqueeze(1).unsqueeze(2)


def create_subsequent_mask(size: int, device: torch.device | None = None) -> torch.Tensor:
    """causal mask (1, 1, T, T) を作る。下三角 True=見てよい（未来 token を遮断）。"""
    mask = torch.tril(torch.ones(size, size, dtype=torch.bool, device=device))
    return mask.view(1, 1, size, size)


class Seq2SeqTransformer(nn.Module):
    """完全な Encoder-Decoder Transformer

    forward(src, tgt_input, src_pad_mask, tgt_mask) → logits (B, T_tgt, tgt_vocab_size)。
    src_pad_mask は encoder self-attn と decoder cross-attn の両方で再利用する
    （どちらも src の pad を参照させない、という同じ要件）。
    """

    def __init__(self, config: Seq2SeqConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        src: torch.Tensor,
        tgt_input: torch.Tensor,
        src_pad_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        memory = self.encoder(src, src_pad_mask)
        logits = self.decoder(tgt_input, memory, tgt_mask, src_pad_mask)
        return logits

    def encode(self, src: torch.Tensor, src_pad_mask: torch.Tensor | None) -> torch.Tensor:
        return self.encoder(src, src_pad_mask)

    @torch.no_grad()
    def greedy_decode(
        self,
        src: torch.Tensor,
        bos_id: int,
        eos_id: int,
        pad_id: int,
        max_len: int,
    ) -> torch.Tensor:
        """<bos> から <eos> まで自己回帰生成（1 バッチ＝1 系列を想定）。"""
        self.eval()
        device = src.device
        src_pad_mask = create_padding_mask(src, pad_id)
        memory = self.encode(src, src_pad_mask)

        ys = torch.full((src.size(0), 1), bos_id, dtype=torch.long, device=device)
        for _ in range(max_len - 1):
            tgt_mask = create_subsequent_mask(ys.size(1), device)
            logits = self.decoder(ys, memory, tgt_mask, src_pad_mask)
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_id], dim=1)
            if bool((next_id == eos_id).all()):
                break
        return ys

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# === 動作確認 ===
# 直接実行: uv run python seq2seq.py
if __name__ == "__main__":
    print("=== Encoder-Decoder Transformer 構造の動作確認 ===\n")

    config = Seq2SeqConfig(
        src_vocab_size=20,
        tgt_vocab_size=20,
        max_len=16,
        n_layer=2,
        n_head=2,
        d_model=64,
        dropout=0.0,
    )
    model = Seq2SeqTransformer(config)
    print(f"パラメータ数: {model.num_parameters():,}")

    B, T_src, T_tgt = 4, 7, 6
    pad_id = 0
    src = torch.randint(1, config.src_vocab_size, (B, T_src))
    tgt_input = torch.randint(1, config.tgt_vocab_size, (B, T_tgt))
    # 末尾を pad にして padding mask が効くことを確認
    src[:, -2:] = pad_id

    src_pad_mask = create_padding_mask(src, pad_id)  # (B, 1, 1, T_src)
    tgt_pad_mask = create_padding_mask(tgt_input, pad_id)  # (B, 1, 1, T_tgt)
    causal = create_subsequent_mask(T_tgt)  # (1, 1, T_tgt, T_tgt)
    # tgt_mask = causal ∧ padding（未来遮断と pad 除外を同時に課す）
    tgt_mask = causal & tgt_pad_mask

    print("\n=== マスク shape ===")
    print(f"src_pad_mask = {tuple(src_pad_mask.shape)}  (B, 1, 1, T_src)")
    print(f"causal       = {tuple(causal.shape)}  (1, 1, T_tgt, T_tgt)")
    print(f"tgt_mask     = {tuple(tgt_mask.shape)}  (B, 1, T_tgt, T_tgt) = causal & tgt_pad")

    print("\n=== Forward の動作確認 ===")
    logits = model(src, tgt_input, src_pad_mask, tgt_mask)
    print(f"src.shape       = {tuple(src.shape)}  (B, T_src)")
    print(f"tgt_input.shape = {tuple(tgt_input.shape)}  (B, T_tgt)")
    print(f"logits.shape    = {tuple(logits.shape)}  (B, T_tgt, tgt_vocab_size)")
    assert logits.shape == (B, T_tgt, config.tgt_vocab_size)

    print("\n=== greedy_decode の動作確認（学習前なのでランダム出力）===")
    out = model.greedy_decode(src, bos_id=1, eos_id=2, pad_id=0, max_len=10)
    print(f"生成 token IDs: {out[0].tolist()}")
    print(f"形: {tuple(out.shape)}  (B, 生成長)")
    print("\nself-test 通過")
