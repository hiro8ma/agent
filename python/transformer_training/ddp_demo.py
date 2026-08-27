"""データ並列（DDP）が本当に勾配を平均しているかを対照実験で確かめる。

DDP を付けても付けなくても学習は進む。損失は下がるし、エラーも出ない。
だから「動いた」ことは勾配が平均されている証明にならない。

このデモは、同じ初期値・別々のデータを各ランクに与えて逆伝播し、
ランク間で勾配が一致するかどうかを見る。

    DDP なし  各ランクは自分のデータぶんの勾配を持つ  → ランク間で食い違う
    DDP あり  all-reduce で平均される                → ランク間で一致する
              かつ、その値は DDP なしの平均と等しい

CUDA が無い環境でも動く。バックエンドに gloo を使い、CPU プロセスを並べる。
教材は nccl を指定しているが、あれは GPU 間通信のライブラリで CUDA が要る。

    uv run python ddp_demo.py
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

IN_FEATURES = 4
N_SAMPLES = 32


def make_dataset() -> TensorDataset:
    """ランクごとに違う勾配が出るよう、値に幅のあるデータを作る。"""
    g = torch.Generator().manual_seed(1234)
    x = torch.randn(N_SAMPLES, IN_FEATURES, generator=g)
    y = torch.randn(N_SAMPLES, 1, generator=g)
    return TensorDataset(x, y)


def worker(rank: int, world_size: int, use_ddp: bool, use_sampler: bool, q) -> None:
    # gloo は CPU 間の集団通信を扱う。nccl は CUDA が要るため、この環境では選べない。
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # 全ランクで同じ初期値にする。ここがずれていると、勾配の比較が意味を持たない。
    torch.manual_seed(0)
    model: nn.Module = nn.Linear(IN_FEATURES, 1, bias=False)
    if use_ddp:
        model = DDP(model)

    dataset = make_dataset()
    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if use_sampler
        else None
    )
    loader = DataLoader(dataset, batch_size=4, sampler=sampler, shuffle=False)

    x, y = next(iter(loader))
    loss = nn.functional.mse_loss(model(x), y)
    loss.backward()

    core = model.module if use_ddp else model
    grad = core.weight.grad.detach().clone().flatten()

    # このランクが実際に見たデータ。DistributedSampler の効果を確かめるために返す。
    seen = [round(float(v), 4) for v in x[:, 0]]

    q.put({"rank": rank, "grad": [round(float(v), 6) for v in grad], "seen": seen})
    dist.destroy_process_group()


def run(world_size: int, use_ddp: bool, use_sampler: bool) -> list[dict]:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    mp.spawn(worker, args=(world_size, use_ddp, use_sampler, q), nprocs=world_size, join=True)
    return sorted((q.get() for _ in range(world_size)), key=lambda r: r["rank"])


def all_equal(rows: list[dict], key: str) -> bool:
    first = rows[0][key]
    return all(r[key] == first for r in rows)


def mean_grad(rows: list[dict]) -> list[float]:
    n = len(rows)
    cols = zip(*(r["grad"] for r in rows))
    return [round(sum(c) / n, 6) for c in cols]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--world-size", type=int, default=4)
    args = p.parse_args()
    n = args.world_size

    print(f"world_size = {n} / backend = gloo / device = cpu\n")

    print("=" * 68)
    print("対照 1  DDP なし — 各ランクは自分のデータぶんの勾配を持つ")
    print("=" * 68)
    baseline = run(n, use_ddp=False, use_sampler=True)
    for r in baseline:
        print(f"  rank{r['rank']}  grad = {r['grad']}")
    same_without = all_equal(baseline, "grad")
    print(f"\n  ランク間で一致するか: {same_without}  ← False であるべき")

    print()
    print("=" * 68)
    print("対照 2  DDP あり — all-reduce で平均される")
    print("=" * 68)
    ddp = run(n, use_ddp=True, use_sampler=True)
    for r in ddp:
        print(f"  rank{r['rank']}  grad = {r['grad']}")
    same_with = all_equal(ddp, "grad")
    print(f"\n  ランク間で一致するか: {same_with}  ← True であるべき")

    print()
    print("=" * 68)
    print("検算  DDP の勾配は、DDP なしの各ランクの平均と一致するか")
    print("=" * 68)
    expect = mean_grad(baseline)
    actual = ddp[0]["grad"]
    # 浮動小数の丸め差を許容する
    matches = all(abs(a - b) < 1e-5 for a, b in zip(expect, actual))
    print(f"  DDP なしの平均 = {expect}")
    print(f"  DDP ありの勾配 = {actual}")
    print(f"  一致: {matches}")

    print()
    print("=" * 68)
    print("おまけ  DistributedSampler を外すと何が起きるか")
    print("=" * 68)
    no_sampler = run(n, use_ddp=True, use_sampler=False)
    for r in no_sampler:
        print(f"  rank{r['rank']}  見たデータの先頭列 = {r['seen']}")
    dup = all_equal(no_sampler, "seen")
    print(f"\n  全ランクが同じデータを見ているか: {dup}  ← True なら並列化できていない")
    print("  エラーは出ない。実質バッチサイズが world_size 倍になるだけで学習は進む。")

    print()
    ok = (not same_without) and same_with and matches and dup
    print("判定:", "勾配平均が働いていることを確認" if ok else "期待どおりの差が出ていない")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
