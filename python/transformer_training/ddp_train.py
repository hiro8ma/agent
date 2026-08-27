"""mini-GPT の学習をデータ並列で回す。

train.py の単一プロセス版に対して、DDP で並列化した版になる。
既存の train.py は壊さず、並列化に必要な差分だけをここに置く。

教材との違いが 2 つある。

第 1 に、バックエンドが gloo になる。教材は nccl を指定しているが、
あれは GPU 間通信のライブラリで CUDA が要る。Apple Silicon では選べない。
gloo なら CPU プロセス間で動くため、手元で挙動を確かめられる。

第 2 に、DistributedSampler を使わない。
この学習は DataLoader ではなく、データ列からランダムなオフセットを切り出す方式で、
ランクごとに乱数の種を変えれば重複しないバッチが得られる。
Sampler はインデックス列を分割する仕組みなので、この方式には噛み合わない。
「教材が Sampler を使っているから」で入れると、機構が合っていないまま動く。

    uv run python ddp_train.py --world-size 4 --max-iters 200
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from data import get_batch, load_data
from model import GPTConfig, MiniGPT


def worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # モデルの初期値は全ランクで揃える。ずれていると DDP が壊れる。
    torch.manual_seed(0)

    train_data, val_data, tokenizer = load_data(args.block_size, device="cpu")

    gpt_config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=0.0,
    )
    model = MiniGPT(gpt_config)
    model = DDP(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # バッチの切り出しはランクごとに別の種を使う。
    # 同じ種だと 4 ランクが同じ位置を読み、実質バッチサイズが 4 倍になるだけになる。
    torch.manual_seed(1000 + rank)

    start = time.time()
    for step in range(args.max_iters):
        x, y = get_batch(train_data, block_size=args.block_size, batch_size=args.batch_size)
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # ここで all-reduce が走り、勾配が平均される
        optimizer.step()

        if rank == 0 and (step % args.log_every == 0 or step == args.max_iters - 1):
            elapsed = time.time() - start
            gbatch = args.batch_size * world_size
            print(
                f"  step {step:>4}  loss {loss.item():.4f}  "
                f"global_batch {gbatch}  {elapsed:.1f}s"
            )

    # 全ランクが同じ重みを持つため、保存は 1 回でよい。
    if rank == 0:
        torch.save(model.module.state_dict(), args.out)
        print(f"\n  保存: {args.out}")

    dist.destroy_process_group()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--world-size", type=int, default=2)
    p.add_argument("--max-iters", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8, help="ランクあたりのバッチ（ローカルバッチ）")
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-layer", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--out", default="ddp_model.pt")
    args = p.parse_args()

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12356")

    print(f"world_size = {args.world_size} / backend = gloo / device = cpu")
    print(f"ローカルバッチ {args.batch_size} × {args.world_size} = "
          f"グローバルバッチ {args.batch_size * args.world_size}\n")

    mp.spawn(worker, args=(args.world_size, args), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
