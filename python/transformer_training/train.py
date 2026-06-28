"""mini-GPT の学習ループ

Phase 1 で説明した数学を 1 ファイルで実装したもの。
- Forward Pass:  model(x, y) で logits と loss を取得
- Loss:          cross-entropy（model 内部で計算）
- Backward Pass: loss.backward() で全パラメータの勾配を計算（autograd）
- Update:        optimizer.step() で重みを更新

学習の進捗:
  step 0:    loss ≈ log(vocab_size) ≈ 4.17  (ランダム初期化、すべて等確率)
  step 200:  loss ≈ 2.5   (英語の文字頻度を学んだ段階)
  step 1000: loss ≈ 2.0   (単語っぽい構造を学んだ段階)
  step 5000: loss ≈ 1.5   (英文っぽい単語列が生成できる)

実行: uv run python train.py
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from bpe import BPETokenizer
from data import DATA_DIR, download_dataset, get_batch, load_data
from model import GPTConfig, MiniGPT

UINT16_MAX = 65535


@dataclass
class TrainConfig:
    """学習のハイパーパラメータ"""

    # データ
    block_size: int = 128  # context 長
    batch_size: int = 32  # 1 ステップで処理するサンプル数
    tokenizer: str = "char"  # char / bpe
    bpe_vocab_size: int = 300  # BPE mode の target vocab size（素朴実装でも短時間で試せる）

    # モデル
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1

    # 学習
    max_iters: int = 3000  # 学習ステップ数
    eval_interval: int = 200  # 何ステップごとに val loss を計算するか
    eval_iters: int = 50  # val loss 計算時のバッチ数
    learning_rate: float = 3e-4  # AdamW の lr (GPT-2 と同じ)
    weight_decay: float = 0.1
    grad_clip: float = 1.0  # gradient clipping のノルム上限
    warmup_iters: int = 100  # learning rate warmup（最初に lr を 0 → max まで線形に上げる）

    # 出力
    save_interval: int = 1000  # checkpoint 保存間隔
    sample_interval: int = 500  # 学習途中で生成サンプルを出力する間隔
    sample_tokens: int = 200  # サンプル生成時の token 数

    # ハードウェア
    device: str = "auto"  # auto / cpu / mps (Apple Silicon GPU) / cuda


def parse_args() -> TrainConfig:
    """CLI 引数を TrainConfig に反映する。

    既存の `uv run python train.py` は char-level のまま動かし、教材の token ID/bin 化を
    試したい時だけ `--tokenizer bpe` を指定する。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", choices=("char", "bpe"), default=TrainConfig.tokenizer)
    parser.add_argument("--bpe-vocab-size", type=int, default=TrainConfig.bpe_vocab_size)
    parser.add_argument("--max-iters", type=int, default=TrainConfig.max_iters)
    parser.add_argument("--eval-interval", type=int, default=TrainConfig.eval_interval)
    parser.add_argument("--eval-iters", type=int, default=TrainConfig.eval_iters)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--block-size", type=int, default=TrainConfig.block_size)
    parser.add_argument("--device", default=TrainConfig.device)
    args = parser.parse_args()

    config = TrainConfig()
    config.tokenizer = args.tokenizer
    config.bpe_vocab_size = args.bpe_vocab_size
    config.max_iters = args.max_iters
    config.eval_interval = args.eval_interval
    config.eval_iters = args.eval_iters
    config.batch_size = args.batch_size
    config.block_size = args.block_size
    config.device = args.device
    return config


def get_device(preference: str) -> str:
    """利用可能な最速デバイスを選ぶ"""
    if preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"  # Apple Silicon GPU
    return "cpu"


def get_lr(step: int, config: TrainConfig) -> float:
    """Linear warmup → 一定 のシンプルなスケジュール

    実用 LLM は cosine decay が標準だが、教育目的では warmup だけで十分。
    """
    if step < config.warmup_iters:
        return config.learning_rate * (step + 1) / config.warmup_iters
    return config.learning_rate


def _bpe_dtype(vocab_size: int) -> type[np.unsignedinteger]:
    """BPE token ID を保持する最小の符号なし整数型を選ぶ。"""
    if vocab_size - 1 <= UINT16_MAX:
        return np.uint16
    return np.uint32


def _bpe_paths(vocab_size: int) -> tuple[Path, Path, Path]:
    """target vocab ごとに tokenizer/bin を分け、古い生成物との衝突を避ける。"""
    stem = f"bpe_tinyshakespeare_vocab{vocab_size}"
    return (
        DATA_DIR / f"{stem}.json",
        DATA_DIR / f"train_{stem}.bin",
        DATA_DIR / f"val_{stem}.bin",
    )


def _prepare_bpe_bins(config: TrainConfig) -> BPETokenizer:
    """BPE tokenizer と train/val .bin を必要なら生成する。"""
    tokenizer_path, train_path, val_path = _bpe_paths(config.bpe_vocab_size)
    if tokenizer_path.exists() and train_path.exists() and val_path.exists():
        return BPETokenizer.load(tokenizer_path)

    text = download_dataset()
    tokenizer = BPETokenizer()
    tokenizer.train(text, target_vocab_size=config.bpe_vocab_size)
    tokenizer.save(tokenizer_path)

    dtype = _bpe_dtype(tokenizer.vocab_size)
    n = int(0.9 * len(text))
    splits = {
        train_path: text[:n],
        val_path: text[n:],
    }
    for path, chunk in splits.items():
        ids = tokenizer.encode(chunk)
        arr = np.array(ids, dtype=dtype)
        path.parent.mkdir(exist_ok=True)
        arr.tofile(path)

    return tokenizer


def load_bpe_data(
    config: TrainConfig,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, BPETokenizer]:
    """BPE/bin 化済みの token ID 列をロードする。

    初回は raw text から tokenizer を学習し、train/val .bin を生成する。以降は np.fromfile
    で即ロードできる。
    """
    tokenizer = _prepare_bpe_bins(config)
    _, train_path, val_path = _bpe_paths(config.bpe_vocab_size)
    dtype = _bpe_dtype(tokenizer.vocab_size)

    train_arr = np.fromfile(train_path, dtype=dtype)
    val_arr = np.fromfile(val_path, dtype=dtype)
    train_data = torch.tensor(train_arr.astype(np.int64), dtype=torch.long, device=device)
    val_data = torch.tensor(val_arr.astype(np.int64), dtype=torch.long, device=device)

    print("[data] tokenizer  = bpe")
    print(f"[data] vocab_size = {tokenizer.vocab_size}")
    print(f"[data] train file = {train_path.name}")
    print(f"[data] val file   = {val_path.name}")
    print(f"[data] train tokens = {len(train_data):,}")
    print(f"[data] val tokens   = {len(val_data):,}")
    print(f"[data] block_size   = {config.block_size}")
    return train_data, val_data, tokenizer


def load_training_data(
    config: TrainConfig,
    device: str,
):
    """設定に応じて char-level または BPE/bin のデータをロードする。"""
    if config.tokenizer == "char":
        return load_data(config.block_size, device=device)
    if config.tokenizer == "bpe":
        return load_bpe_data(config, device=device)
    raise ValueError(f"unsupported tokenizer: {config.tokenizer}")


@torch.no_grad()
def estimate_loss(
    model: MiniGPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    config: TrainConfig,
) -> dict[str, float]:
    """train / val 両方の loss を平均で評価（ノイズを減らす）"""
    out = {}
    model.eval()
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            x, y = get_batch(data, block_size=config.block_size, batch_size=config.batch_size)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main() -> None:
    config = parse_args()
    device = get_device(config.device)

    print("=== mini-GPT 学習開始 ===")
    print(f"device         = {device}")
    print(f"tokenizer      = {config.tokenizer}")
    if config.tokenizer == "bpe":
        print(f"bpe_vocab_size = {config.bpe_vocab_size}")
    print(f"max_iters      = {config.max_iters}")
    print(f"batch_size     = {config.batch_size}")
    print(f"block_size     = {config.block_size}")
    print(f"learning_rate  = {config.learning_rate}")
    print()

    # データ準備
    train_data, val_data, tokenizer = load_training_data(config, device=device)

    # モデル構築
    gpt_config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=config.block_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        dropout=config.dropout,
    )
    model = MiniGPT(gpt_config).to(device)
    print(f"\n[model] パラメータ数 = {model.num_parameters():,}")

    # AdamW Optimizer（現代 LLM の標準）
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),  # GPT-3 と同じ
    )

    # 学習ループ本体
    print("\n=== 学習ループ開始 ===")
    t_start = time.time()

    for step in range(config.max_iters):
        # ---- learning rate scheduling ----
        lr = get_lr(step, config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # ---- 評価（定期的に） ----
        if step % config.eval_interval == 0 or step == config.max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, config)
            elapsed = time.time() - t_start
            print(
                f"step {step:5d} | "
                f"train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f} | "
                f"lr {lr:.5f} | "
                f"elapsed {elapsed:6.1f}s"
            )

        # ---- 学習ステップ ----
        # 1. Forward
        x, y = get_batch(train_data, block_size=config.block_size, batch_size=config.batch_size)
        _, loss = model(x, y)

        # 2. Backward (autograd で全パラメータの勾配を計算)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # 3. Gradient clipping (爆発防止)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        # 4. Update (重みを少し動かす)
        optimizer.step()

        # ---- 学習途中での生成サンプル ----
        if step > 0 and step % config.sample_interval == 0:
            print(f"\n--- step {step} の生成サンプル ---")
            model.eval()
            start = torch.zeros((1, 1), dtype=torch.long, device=device)
            with torch.no_grad():
                generated = model.generate(
                    start,
                    max_new_tokens=config.sample_tokens,
                    temperature=0.8,
                )
            text = tokenizer.decode(generated[0].tolist())
            print(text)
            print("--- end sample ---\n")
            model.train()

        # ---- checkpoint 保存 ----
        if step > 0 and step % config.save_interval == 0:
            save_checkpoint(model, gpt_config, tokenizer, step, config.tokenizer)

    # 最終 checkpoint 保存
    save_checkpoint(model, gpt_config, tokenizer, config.max_iters, config.tokenizer)

    # 最終生成サンプル
    print("\n=== 最終生成サンプル ===")
    model.eval()
    start = torch.zeros((1, 1), dtype=torch.long, device=device)
    with torch.no_grad():
        generated = model.generate(start, max_new_tokens=500, temperature=0.8, top_k=40)
    print(tokenizer.decode(generated[0].tolist()))


def save_checkpoint(
    model: MiniGPT,
    gpt_config: GPTConfig,
    tokenizer,
    step: int,
    tokenizer_type: str,
) -> None:
    """checkpoint を保存（モデル重み + 設定 + tokenizer の vocab）"""
    ckpt_dir = Path(__file__).parent / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    path = ckpt_dir / f"mini_gpt_step{step}.pt"
    tokenizer_state: dict[str, Any]
    if tokenizer_type == "char":
        tokenizer_state = {
            "type": "char",
            "char_to_id": tokenizer.char_to_id,
        }
    elif tokenizer_type == "bpe":
        tokenizer_state = {
            "type": "bpe",
            "vocab_size": tokenizer.vocab_size,
            "merges": [[list(pair), new_id] for pair, new_id in tokenizer.merges],
            "end_token": tokenizer.end_token,
            "end_token_id": tokenizer.end_token_id,
        }
    else:
        raise ValueError(f"unsupported tokenizer_type: {tokenizer_type}")

    torch.save(
        {
            "model_state": model.state_dict(),
            "config": gpt_config,
            "tokenizer": tokenizer_state,
            # 後方互換: 既存 checkpoint を generate.py で読めるように char_to_id も残す
            "char_to_id": tokenizer.char_to_id if tokenizer_type == "char" else None,
            "step": step,
        },
        path,
    )
    print(f"[checkpoint] saved to {path}")


if __name__ == "__main__":
    main()
