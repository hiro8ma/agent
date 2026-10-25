from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def rank(scores: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-scores, axis=-1, kind="stable")[..., :k]


def ndcg_at_k(ranked: Sequence[str], qrels: dict[str, int], k: int = 10) -> float:
    dcg = sum(qrels.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked: Sequence[str], qrels: dict[str, int], k: int = 100) -> float:
    relevant = {d for d, r in qrels.items() if r > 0}
    if not relevant:
        return 0.0
    return len(relevant & set(ranked[:k])) / len(relevant)


def overlap_at_k(a: Sequence[str], b: Sequence[str], k: int = 10) -> float:
    return len(set(a[:k]) & set(b[:k])) / k
