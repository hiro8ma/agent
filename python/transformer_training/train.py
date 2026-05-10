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

import time
from dataclasses import dataclass
from pathlib import Path

import torch

from data import get_batch, load_data
from model import GPTConfig, MiniGPT


@dataclass
class TrainConfig:
    """学習のハイパーパラメータ"""

    # データ
    block_size: int = 128  # context 長
    batch_size: int = 32  # 1 ステップで処理するサンプル数

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
    config = TrainConfig()
    device = get_device(config.device)

    print(f"=== mini-GPT 学習開始 ===")
    print(f"device         = {device}")
    print(f"max_iters      = {config.max_iters}")
    print(f"batch_size     = {config.batch_size}")
    print(f"block_size     = {config.block_size}")
    print(f"learning_rate  = {config.learning_rate}")
    print()

    # データ準備
    train_data, val_data, tokenizer = load_data(config.block_size, device=device)

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
    print(f"\n=== 学習ループ開始 ===")
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
                generated = model.generate(start, max_new_tokens=config.sample_tokens, temperature=0.8)
            text = tokenizer.decode(generated[0].tolist())
            print(text)
            print("--- end sample ---\n")
            model.train()

        # ---- checkpoint 保存 ----
        if step > 0 and step % config.save_interval == 0:
            save_checkpoint(model, gpt_config, tokenizer, step)

    # 最終 checkpoint 保存
    save_checkpoint(model, gpt_config, tokenizer, config.max_iters)

    # 最終生成サンプル
    print(f"\n=== 最終生成サンプル ===")
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
) -> None:
    """checkpoint を保存（モデル重み + 設定 + tokenizer の vocab）"""
    ckpt_dir = Path(__file__).parent / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    path = ckpt_dir / f"mini_gpt_step{step}.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": gpt_config,
            "char_to_id": tokenizer.char_to_id,
            "step": step,
        },
        path,
    )
    print(f"[checkpoint] saved to {path}")


if __name__ == "__main__":
    main()
