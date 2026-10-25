from __future__ import annotations

import numpy as np


def normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.asarray(x / np.maximum(norm, 1e-12))


def interpolate(query: np.ndarray, generated: np.ndarray, alpha: float) -> np.ndarray:
    return alpha * normalize(query) + (1 - alpha) * normalize(generated)


def mean_with_query(query: np.ndarray, generated: list[np.ndarray]) -> np.ndarray:
    stacked = np.stack([normalize(query), *(normalize(g) for g in generated)])
    return np.asarray(stacked.mean(axis=0))


def scores(doc_emb: np.ndarray, query_vecs: np.ndarray) -> np.ndarray:
    return np.asarray(query_vecs @ doc_emb.T)
