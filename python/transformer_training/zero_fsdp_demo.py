"""ZeRO（FSDP）が本当にパラメータを分散保持しているかを実測する。

教材の主張は「各 GPU が実際にメモリ上に持つパラメータ数が少なくなる」。
DDP と違い、モデルの複製を各ランクが持たない。

これは数えれば確かめられる。ランクごとに、自分が保持しているパラメータ要素数を
足し上げ、DDP の場合と比べる。

    DDP   各ランクが全パラメータを持つ        → 合計 = P × world_size
    FSDP  各ランクは 1/world_size だけ持つ    → 合計 ≒ P

学習が進むこと自体は両方で起きるため、「動いた」では区別できない。
持っている量を数えるのが唯一の判別方法になる。

    uv run python zero_fsdp_demo.py --world-size 2
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.nn.parallel import DistributedDataParallel as DDP


class Net(nn.Module):
    """層をいくつか重ねただけのモデル。FSDP は層ごとに分割単位を作る。"""

    def __init__(self, width: int = 256, depth: int = 4) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers += [nn.Linear(width, width), nn.ReLU()]
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))


def local_param_elements(model: nn.Module) -> int:
    """このランクが実際に手元に持っているパラメータ要素数を数える。

    FSDP のパラメータは DTensor になり、.to_local() でこのランクの分片が取れる。
    通常の Tensor ならそのまま全体を持っている。
    """
    total = 0
    for p in model.parameters():
        local = p.to_local() if hasattr(p, "to_local") else p
        total += local.numel()
    return total


def worker(rank: int, world_size: int, mode: str, q) -> None:
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.manual_seed(0)

    model: nn.Module = Net()
    full = sum(p.numel() for p in model.parameters())

    if mode == "ddp":
        model = DDP(model)
    elif mode == "fsdp":
        # デバイスメッシュを CPU で明示する。
        # 省略すると既定デバイスの判定が走り、Apple Silicon では MPS と見なされて
        # torch.mps.is_initialized が無いために落ちる。
        mesh = init_device_mesh("cpu", (world_size,))
        # 層ごとに分割単位を作ってから、全体を包む。
        for layer in model.body:
            if isinstance(layer, nn.Linear):
                fully_shard(layer, mesh=mesh)
        fully_shard(model, mesh=mesh)

    held = local_param_elements(model)

    # 学習が両方で進むことも確かめる。動くだけでは区別できない、という点の裏づけ。
    x = torch.randn(8, 256)
    y = torch.randn(8, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    first = last = 0.0
    for step in range(5):
        loss = nn.functional.mse_loss(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()

    q.put({"rank": rank, "mode": mode, "full": full, "held": held,
           "first": round(first, 4), "last": round(last, 4)})
    dist.destroy_process_group()


def run(world_size: int, mode: str) -> list[dict]:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12357")
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    mp.spawn(worker, args=(world_size, mode, q), nprocs=world_size, join=True)
    return sorted((q.get() for _ in range(world_size)), key=lambda r: r["rank"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--world-size", type=int, default=2)
    args = p.parse_args()
    n = args.world_size

    print(f"world_size = {n} / backend = gloo / device = cpu\n")

    results = {}
    for mode in ("ddp", "fsdp"):
        rows = run(n, mode)
        results[mode] = rows
        full = rows[0]["full"]
        held_sum = sum(r["held"] for r in rows)
        print("=" * 68)
        print(f"{mode.upper()}  モデル全体のパラメータ = {full:,}")
        print("=" * 68)
        for r in rows:
            ratio = r["held"] / full
            print(f"  rank{r['rank']}  保持 {r['held']:>9,} 要素  "
                  f"（全体の {ratio:.0%}）  loss {r['first']} -> {r['last']}")
        print(f"  全ランク合計 {held_sum:,} 要素  = 全体の {held_sum / full:.1f} 倍\n")

    full = results["ddp"][0]["full"]
    ddp_sum = sum(r["held"] for r in results["ddp"])
    fsdp_sum = sum(r["held"] for r in results["fsdp"])

    print("=" * 68)
    print("判定")
    print("=" * 68)
    print(f"  DDP  合計 {ddp_sum:,} = 全体 × {ddp_sum / full:.1f}  （各ランクが複製を持つ）")
    print(f"  FSDP 合計 {fsdp_sum:,} = 全体 × {fsdp_sum / full:.1f}  （分散保持）")
    print(f"  1 ランクあたりの削減: {1 - (fsdp_sum / n) / full:.0%}")
    print()
    print("  どちらも loss は下がる。学習が進むことは分散保持の証明にならない。")

    ok = ddp_sum >= full * n * 0.99 and fsdp_sum <= full * 1.5
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
