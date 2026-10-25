from __future__ import annotations

import argparse
import math
import platform
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
import torch

from data import (
    Catalog,
    Pairs,
    count_overlapping_rows,
    make_catalog,
    mutual_information,
    sample_pairs,
)
from losses import candidate_log_q, contrastive_loss, info_nce_bound
from model import TwoTower

ScoreFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class Config:
    n_items: int = 20_000
    feature_dim: int = 64
    n_clusters: int = 200
    zipf_s: float = 1.0
    noise: float = 1.0
    top_fraction: float = 0.01
    n_train: int = 300_000
    n_test: int = 20_000
    hidden: int = 256
    embed_dim: int = 32
    temperature: float = 0.05
    batch_size: int = 1024
    epochs: int = 8
    lr: float = 3e-3
    data_seed: int = 0
    seeds: tuple[int, ...] = (0, 1, 2)
    overlap_batch_sizes: tuple[int, ...] = (256, 1024, 4096)
    bound_negatives: tuple[int, ...] = (2, 8, 32, 128, 512, 2048, 8192)


QUICK = Config(
    n_items=2_000,
    n_clusters=40,
    n_train=20_000,
    n_test=2_000,
    batch_size=256,
    epochs=2,
    seeds=(0,),
    overlap_batch_sizes=(64, 256),
    bound_negatives=(2, 16, 128),
)


@dataclass(frozen=True)
class Variant:
    key: str
    label: str
    uniform: bool
    log_q: bool
    remove_false_negatives: bool


VARIANTS = (
    Variant("a", "バッチ内負例だけ", uniform=False, log_q=False, remove_false_negatives=False),
    Variant("b", "a + 一様な負例", uniform=True, log_q=False, remove_false_negatives=False),
    Variant("c", "b + logQ 補正", uniform=True, log_q=True, remove_false_negatives=False),
    Variant("d", "c + 偽の負例の除外", uniform=True, log_q=True, remove_false_negatives=True),
)


@dataclass(frozen=True)
class Metrics:
    recall_10: float
    recall_50: float
    popular_share_10: float


def positive_frequency(pairs: Pairs, n_items: int) -> torch.Tensor:
    counts = np.bincount(pairs.positive, minlength=n_items)
    return torch.from_numpy(counts / counts.sum()).float()


def popular_items(freq: torch.Tensor, fraction: float) -> torch.Tensor:
    mask = torch.zeros(freq.numel(), dtype=torch.bool)
    mask[freq.topk(max(1, round(freq.numel() * fraction))).indices] = True
    return mask


def train(
    cfg: Config, catalog: Catalog, pairs: Pairs, freq: torch.Tensor, variant: Variant, seed: int
) -> tuple[TwoTower, float]:
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = TwoTower(torch.from_numpy(catalog.features), cfg.hidden, cfg.embed_dim)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    anchor = torch.from_numpy(pairs.anchor)
    positive = torch.from_numpy(pairs.positive)
    n_uniform = cfg.batch_size if variant.uniform else 0
    log_q = candidate_log_q(freq, cfg.batch_size, n_uniform) if variant.log_q else None
    n_batches = len(pairs) // cfg.batch_size
    last_epoch_loss = 0.0
    for _ in range(cfg.epochs):
        order = torch.randperm(len(pairs), generator=gen)
        total = 0.0
        for b in range(n_batches):
            idx = order[b * cfg.batch_size : (b + 1) * cfg.batch_size]
            pos_ids = positive[idx]
            candidates = pos_ids
            if n_uniform:
                uniform = torch.randint(catalog.n_items, (n_uniform,), generator=gen)
                candidates = torch.cat([pos_ids, uniform])
            scores = model.query(anchor[idx]) @ model.item(candidates).T / cfg.temperature
            loss = contrastive_loss(
                scores, pos_ids, candidates, log_q, variant.remove_false_negatives
            )
            opt.zero_grad()
            torch.autograd.backward(loss)
            opt.step()
            total += loss.item()
        last_epoch_loss = total / n_batches
    return model, last_epoch_loss


def model_scores(model: TwoTower, n_items: int) -> ScoreFn:
    with torch.no_grad():
        items = model.item(torch.arange(n_items))

    def score(anchor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return model.query(anchor) @ items.T

    return score


def oracle_scores(catalog: Catalog) -> ScoreFn:
    log_pop = torch.from_numpy(np.log(catalog.popularity)).float()
    cluster = torch.from_numpy(catalog.cluster)

    def score(anchor: torch.Tensor) -> torch.Tensor:
        same = cluster[anchor].unsqueeze(1) == cluster.unsqueeze(0)
        return log_pop.unsqueeze(0).masked_fill(~same, float("-inf"))

    return score


def popularity_scores(freq: torch.Tensor) -> ScoreFn:
    def score(anchor: torch.Tensor) -> torch.Tensor:
        return freq.unsqueeze(0).expand(anchor.numel(), -1).clone()

    return score


def evaluate(score_fn: ScoreFn, pairs: Pairs, popular: torch.Tensor, chunk: int = 1024) -> Metrics:
    anchor = torch.from_numpy(pairs.anchor)
    positive = torch.from_numpy(pairs.positive)
    hit_10 = hit_50 = 0
    share = 0.0
    for start in range(0, len(pairs), chunk):
        a = anchor[start : start + chunk]
        p = positive[start : start + chunk].unsqueeze(1)
        scores = score_fn(a)
        scores[torch.arange(a.numel()), a] = float("-inf")
        top = scores.topk(50, dim=1).indices
        hit_10 += int((top[:, :10] == p).any(dim=1).sum())
        hit_50 += int((top == p).any(dim=1).sum())
        share += float(popular[top[:, :10]].float().mean(dim=1).sum())
    n = len(pairs)
    return Metrics(hit_10 / n, hit_50 / n, share / n)


def overlap_rows(
    pairs: Pairs, batch_sizes: tuple[int, ...], n_items: int, seed: int
) -> list[tuple[int, float, float]]:
    rng = np.random.default_rng(seed)
    rows = []
    for batch in batch_sizes:
        order = rng.permutation(len(pairs))
        in_batch, uniform_hit = [], []
        for b in range(len(pairs) // batch):
            pos = pairs.positive[order[b * batch : (b + 1) * batch]]
            in_batch.append(count_overlapping_rows(pos))
            uniform_hit.append(int(np.isin(pos, rng.integers(n_items, size=batch)).sum()))
        rows.append((batch, float(np.mean(in_batch)), float(np.mean(uniform_hit))))
    return rows


def bound_curve(
    model: TwoTower,
    pairs: Pairs,
    freq: torch.Tensor,
    negatives: tuple[int, ...],
    temperature: float,
    seed: int,
    chunk: int = 500,
) -> dict[int, float]:
    gen = torch.Generator().manual_seed(seed)
    cdf = torch.cumsum(freq.double(), dim=0)
    cdf = cdf / cdf[-1]
    score = model_scores(model, freq.numel())
    anchor = torch.from_numpy(pairs.anchor)
    positive = torch.from_numpy(pairs.positive)
    curve = {}
    for n in negatives:
        total = 0.0
        for start in range(0, len(pairs), chunk):
            a = anchor[start : start + chunk]
            logits = score(a) / temperature
            pos = logits.gather(1, positive[start : start + chunk].unsqueeze(1)).squeeze(1)
            u = torch.rand(a.numel(), n - 1, generator=gen, dtype=torch.float64)
            neg_ids = torch.searchsorted(cdf, u).clamp_(max=freq.numel() - 1)
            total += float(info_nce_bound(pos, logits.gather(1, neg_ids))) * a.numel()
        curve[n] = total / len(pairs)
    return curve


def fmt(values: list[float]) -> str:
    if len(values) == 1:
        return f"{values[0]:.3f}"
    return f"{np.mean(values):.3f} ± {np.std(values):.3f}"


def run_seeds(
    cfg: Config,
    catalog: Catalog,
    train_pairs: Pairs,
    test_pairs: Pairs,
    freq: torch.Tensor,
    popular: torch.Tensor,
    variant: Variant,
) -> tuple[list[Metrics], TwoTower]:
    metrics, models = [], []
    for seed in cfg.seeds:
        t0 = time.perf_counter()
        model, loss = train(cfg, catalog, train_pairs, freq, variant, seed)
        m = evaluate(model_scores(model, catalog.n_items), test_pairs, popular)
        metrics.append(m)
        models.append(model)
        print(
            f"[{variant.key} batch={cfg.batch_size} seed={seed}] loss {loss:.3f} "
            f"recall@10 {m.recall_10:.4f} recall@50 {m.recall_50:.4f} "
            f"popular {m.popular_share_10:.4f} {time.perf_counter() - t0:.1f}s",
            file=sys.stderr,
        )
    return metrics, models[0]


def metric_cells(ms: list[Metrics]) -> str:
    return (
        f"{fmt([m.recall_10 for m in ms])} | {fmt([m.recall_50 for m in ms])} "
        f"| {fmt([m.popular_share_10 for m in ms])}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="compare negative handling in InfoNCE")
    parser.add_argument("--quick", action="store_true", help="tiny config for a smoke run")
    args = parser.parse_args()
    cfg = QUICK if args.quick else Config()
    started = time.perf_counter()

    print(
        f"python {platform.python_version()} / torch {torch.__version__} / numpy {np.__version__}"
    )
    print(f"platform {platform.platform()} / torch threads {torch.get_num_threads()}")
    print(cfg)

    rng = np.random.default_rng(cfg.data_seed)
    catalog = make_catalog(rng, cfg.n_items, cfg.feature_dim, cfg.n_clusters, cfg.zipf_s, cfg.noise)
    pool = sample_pairs(rng, catalog, cfg.n_train + cfg.n_test)
    train_pairs = Pairs(pool.anchor[: cfg.n_train], pool.positive[: cfg.n_train])
    test_pairs = Pairs(pool.anchor[cfg.n_train :], pool.positive[cfg.n_train :])
    freq = positive_frequency(train_pairs, cfg.n_items)
    popular = popular_items(freq, cfg.top_fraction)
    true_mi = mutual_information(catalog)

    print("\n## data")
    print(f"items seen as a training positive {int((freq > 0).sum())} / {cfg.n_items}")
    print(f"test positives in the top 1% {popular[test_pairs.positive].float().mean():.3f}")
    print(f"true I(anchor; positive) {true_mi:.3f} nats")

    print("\n## overlap\n")
    print("| バッチサイズ | 正例が重なる行数（平均） | 割合 | 一様な負例に正例が入る行数（平均） |")
    print("|---:|---:|---:|---:|")
    for batch, rows, uniform_hit in overlap_rows(
        train_pairs, cfg.overlap_batch_sizes, cfg.n_items, cfg.data_seed
    ):
        print(f"| {batch:,} | {rows:.1f} | {rows / batch:.3f} | {uniform_hit:.1f} |")

    results: dict[str, list[Metrics]] = {}
    first_models: dict[str, TwoTower] = {}
    for variant in VARIANTS:
        results[variant.key], first_models[variant.key] = run_seeds(
            cfg, catalog, train_pairs, test_pairs, freq, popular, variant
        )

    print(f"\n## retrieval (batch {cfg.batch_size:,}, seeds {list(cfg.seeds)})\n")
    print("| 変種 | recall@10 | recall@50 | 上位10件に占める人気上位1%の割合 |")
    print("|---|---:|---:|---:|")
    for variant in VARIANTS:
        print(f"| {variant.key}. {variant.label} | {metric_cells(results[variant.key])} |")
    for label, fn in (
        ("参考 人気順", popularity_scores(freq)),
        ("参考 真の条件付き分布", oracle_scores(catalog)),
    ):
        m = evaluate(fn, test_pairs, popular)
        print(f"| {label} | {m.recall_10:.3f} | {m.recall_50:.3f} | {m.popular_share_10:.3f} |")

    print(f"\n## false negative removal by batch size (seeds {list(cfg.seeds)})\n")
    print("| バッチサイズ | 変種 | recall@10 | recall@50 | 上位10件に占める人気上位1%の割合 |")
    print("|---:|---|---:|---:|---:|")
    for batch in cfg.overlap_batch_sizes:
        for variant in VARIANTS[2:]:
            if batch == cfg.batch_size:
                ms = results[variant.key]
            else:
                ms, _ = run_seeds(
                    replace(cfg, batch_size=batch),
                    catalog,
                    train_pairs,
                    test_pairs,
                    freq,
                    popular,
                    variant,
                )
            print(f"| {batch:,} | {variant.key} | {metric_cells(ms)} |")

    print(f"\n## InfoNCE bound (seed {cfg.seeds[0]}, {len(test_pairs):,} test pairs)\n")
    curves = {
        key: bound_curve(
            model, test_pairs, freq, cfg.bound_negatives, cfg.temperature, cfg.data_seed
        )
        for key, model in first_models.items()
    }
    header = " | ".join(v.key for v in VARIANTS)
    print(f"| 候補数 N | log N | {header} |")
    print("|---:|---:|" + "---:|" * len(VARIANTS))
    for n in cfg.bound_negatives:
        cells = " | ".join(f"{curves[v.key][n]:.3f}" for v in VARIANTS)
        print(f"| {n:,} | {math.log(n):.3f} | {cells} |")
    print(f"\ntrue I(anchor; positive) {true_mi:.3f} nats")
    print(f"\ntotal {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
