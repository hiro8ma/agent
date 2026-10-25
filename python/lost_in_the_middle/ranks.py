from __future__ import annotations

import math
import random
from collections.abc import Sequence


def zipf_weights(k: int) -> list[float]:
    return [1.0 / r for r in range(1, k + 1)]


def uniform_weights(k: int) -> list[float]:
    return [1.0] * k


def allocate(n: int, weights: Sequence[float]) -> list[int]:
    total = sum(weights)
    quotas = [n * w / total for w in weights]
    counts = [math.floor(q) for q in quotas]
    by_remainder = sorted(range(len(weights)), key=lambda i: (counts[i] - quotas[i], i))
    for i in by_remainder[: n - sum(counts)]:
        counts[i] += 1
    return counts


def assign_ranks(n: int, weights: Sequence[float], seed: int) -> list[int]:
    ranks = [r for r, c in enumerate(allocate(n, weights), start=1) for _ in range(c)]
    random.Random(seed).shuffle(ranks)
    return ranks
