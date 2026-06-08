"""Encoder-Decoder Transformer を合成 seq2seq タスクで学習するデモ

外部コーパスのダウンロードは不要。オフラインで動く合成 parallel corpus を生成して
「翻訳」の代わりに **系列反転タスク**（入力 [a,b,c] → 出力 [c,b,a]）を学習させる。
反転は cross-attention が「入力のどの位置を見るか」を学べないと解けない。位置を逆順に
辿る必要があるため、cross-attention の動作確認に向く最小タスク。

学習の要点:
  - teacher forcing  デコーダ入力は <bos>+正解の先頭 n-1、正解は正解+<eos>（1 つずらし）
  - padding          バッチ内最長に <pad> で揃える（collate_fn）
  - ignore_index     loss で <pad> 位置を無視（pad の予測精度は無意味なため）

数百ステップで CPU 上 1 分以内に収束する規模にしてある。
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from seq2seq import (
    Seq2SeqConfig,
    Seq2SeqTransformer,
    create_padding_mask,
    create_subsequent_mask,
)

# 特殊トークン（ID は固定）。pad=0 は create_padding_mask の既定と揃える。
PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIAL = 4  # ここから先が通常記号
N_SYMBOLS = 12  # 通常記号の種類数
VOCAB_SIZE = SPECIAL + N_SYMBOLS

MIN_LEN, MAX_SEQ = 3, 8  # 1 系列の記号数の範囲


def make_pair(rng: random.Random) -> tuple[list[int], list[int]]:
    """1 つの (src, tgt) ペアを作る。tgt は src の反転。"""
    n = rng.randint(MIN_LEN, MAX_SEQ)
    src = [rng.randint(SPECIAL, VOCAB_SIZE - 1) for _ in range(n)]
    tgt = list(reversed(src))
    return src, tgt


class ReverseDataset(Dataset[tuple[list[int], list[int]]]):
    """系列反転タスクの合成データセット"""

    def __init__(self, size: int, seed: int) -> None:
        rng = random.Random(seed)
        self.pairs = [make_pair(rng) for _ in range(size)]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[list[int], list[int]]:
        return self.pairs[idx]


def collate_fn(
    batch: list[tuple[list[int], list[int]]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """バッチ内最長に <pad> で揃える（pad_sequence 相当）。

    src:        各系列に <eos> を付与
    tgt_full:   <bos> + 記号列 + <eos>（後で input/output に 1 つずらして使う）
    """
    srcs = [s + [EOS] for s, _ in batch]
    tgts = [[BOS] + t + [EOS] for _, t in batch]
    src_max = max(len(s) for s in srcs)
    tgt_max = max(len(t) for t in tgts)

    def pad(seqs: list[list[int]], width: int) -> torch.Tensor:
        return torch.tensor([s + [PAD] * (width - len(s)) for s in seqs], dtype=torch.long)

    src = pad(srcs, src_max)
    tgt_full = pad(tgts, tgt_max)
    return src, tgt_full, torch.tensor([len(s) for s in srcs])


def build_masks(
    src: torch.Tensor, tgt_input: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """src padding mask と tgt mask（causal ∧ padding）を作る。"""
    src_pad_mask = create_padding_mask(src, PAD)
    tgt_pad_mask = create_padding_mask(tgt_input, PAD)
    causal = create_subsequent_mask(tgt_input.size(1), tgt_input.device)
    tgt_mask = causal & tgt_pad_mask
    return src_pad_mask, tgt_mask


def train() -> Seq2SeqTransformer:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = Seq2SeqConfig(
        src_vocab_size=VOCAB_SIZE,
        tgt_vocab_size=VOCAB_SIZE,
        max_len=MAX_SEQ + 4,
        n_layer=3,
        n_head=4,
        d_model=128,
        dropout=0.1,
    )
    model = Seq2SeqTransformer(config).to(device)
    print(f"パラメータ数: {model.num_parameters():,}  device={device}")

    train_ds = ReverseDataset(size=4000, seed=1)
    loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn)

    # ignore_index=PAD: pad 位置の予測誤差を loss から除外（teacher forcing の埋め草なため）
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    max_steps = 600
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)

    print("\n=== 学習 ===")
    model.train()
    step = 0
    first_loss = None
    last_loss = 0.0
    while step < max_steps:
        for src, tgt_full, _ in loader:
            src, tgt_full = src.to(device), tgt_full.to(device)
            # teacher forcing: 入力は末尾 <eos> を除く / 正解は先頭 <bos> を除く（1 つずらし）
            tgt_input = tgt_full[:, :-1]
            tgt_output = tgt_full[:, 1:]

            src_pad_mask, tgt_mask = build_masks(src, tgt_input)
            logits = model(src, tgt_input, src_pad_mask, tgt_mask)

            # (B, T, V) → (B*T, V) と (B*T,) に flatten して cross-entropy
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1),
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if first_loss is None:
                first_loss = loss.item()
            last_loss = loss.item()
            step += 1
            if step % 100 == 0:
                print(f"step {step:4d}/{max_steps}  loss {loss.item():.4f}")
            if step >= max_steps:
                break

    assert first_loss is not None
    print(f"\nloss: {first_loss:.4f} → {last_loss:.4f}（低下を確認）")
    return model


@torch.no_grad()
def evaluate(model: Seq2SeqTransformer) -> float:
    """テスト集合で greedy デコードし、完全一致の正解率を返す。"""
    device = next(model.parameters()).device
    test_ds = ReverseDataset(size=300, seed=999)
    correct = 0
    for src_list, tgt_list in test_ds.pairs:
        src = torch.tensor([src_list + [EOS]], dtype=torch.long, device=device)
        out = model.greedy_decode(
            src, bos_id=BOS, eos_id=EOS, pad_id=PAD, max_len=MAX_SEQ + 3
        )
        pred = out[0].tolist()[1:]  # 先頭 <bos> を除く
        if EOS in pred:
            pred = pred[: pred.index(EOS)]
        if pred == tgt_list:
            correct += 1
    return correct / len(test_ds.pairs)


def demo_examples(model: Seq2SeqTransformer, n: int = 3) -> None:
    device = next(model.parameters()).device
    rng = random.Random(7)
    print("\n=== 推論デモ（入力 → 期待 / 生成）===")
    for _ in range(n):
        src_list, tgt_list = make_pair(rng)
        src = torch.tensor([src_list + [EOS]], dtype=torch.long, device=device)
        out = model.greedy_decode(
            src, bos_id=BOS, eos_id=EOS, pad_id=PAD, max_len=MAX_SEQ + 3
        )
        pred = out[0].tolist()[1:]
        if EOS in pred:
            pred = pred[: pred.index(EOS)]
        ok = "OK" if pred == tgt_list else "NG"
        print(f"  [{ok}] {src_list} → 期待 {tgt_list} / 生成 {pred}")


if __name__ == "__main__":
    print("=== Encoder-Decoder Transformer: 系列反転タスクの学習 ===\n")
    model = train()
    acc = evaluate(model)
    demo_examples(model)
    print(f"\nテスト正解率（完全一致）: {acc * 100:.1f}%  (n=300)")
