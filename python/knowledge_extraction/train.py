from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from model import GPT

IGNORE = -100


@dataclass(frozen=True)
class Example:
    ids: tuple[int, ...]
    loss_from: int = 0


def length_grouped_batches(
    lengths: np.ndarray, batch_size: int, rng: np.random.Generator
) -> list[np.ndarray]:
    order = rng.permutation(len(lengths))
    order = order[np.argsort(lengths[order], kind="stable")]
    batches = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
    return [batches[i] for i in rng.permutation(len(batches))]


def collate(
    examples: Sequence[Example], idx: np.ndarray, pad: int
) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(examples[i].ids) for i in idx) - 1
    x = torch.full((len(idx), width), pad, dtype=torch.long)
    y = torch.full((len(idx), width), IGNORE, dtype=torch.long)
    for row, i in enumerate(idx.tolist()):
        ex = examples[i]
        n = len(ex.ids) - 1
        x[row, :n] = torch.tensor(ex.ids[:-1])
        y[row, ex.loss_from : n] = torch.tensor(ex.ids[ex.loss_from + 1 :])
    return x, y


def train(
    model: GPT,
    examples: Sequence[Example],
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    pad: int,
    seed: int,
    warmup_frac: float = 0.02,
    cosine: bool = True,
    on_epoch: Callable[[int, float], None] | None = None,
) -> float:
    rng = np.random.default_rng(seed)
    lengths = np.array([len(e.ids) for e in examples])
    total = epochs * math.ceil(len(examples) / batch_size)
    warmup = max(1, int(total * warmup_frac))
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.98),
    )

    def schedule(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        if not cosine:
            return 1.0
        progress = (step - warmup) / max(1, total - warmup)
        return 0.1 + 0.45 * (1 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, schedule)
    model.train()
    epoch_loss = 0.0
    for epoch in range(epochs):
        losses = []
        for idx in length_grouped_batches(lengths, batch_size, rng):
            x, y = collate(examples, idx, pad)
            logits = model(x)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=IGNORE
            )
            opt.zero_grad(set_to_none=True)
            torch.autograd.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
        epoch_loss = float(np.mean(losses))
        if on_epoch is not None:
            on_epoch(epoch, epoch_loss)
    model.eval()
    return epoch_loss
