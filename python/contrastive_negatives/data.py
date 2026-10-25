from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Catalog:
    features: np.ndarray
    cluster: np.ndarray
    popularity: np.ndarray

    @property
    def n_items(self) -> int:
        return int(self.popularity.shape[0])


@dataclass(frozen=True)
class Pairs:
    anchor: np.ndarray
    positive: np.ndarray

    def __len__(self) -> int:
        return int(self.anchor.shape[0])


def make_catalog(
    rng: np.random.Generator,
    n_items: int,
    dim: int,
    n_clusters: int,
    zipf_s: float,
    noise: float,
) -> Catalog:
    if n_items < 2 * n_clusters:
        raise ValueError("each cluster needs at least two items")
    centers = rng.standard_normal((n_clusters, dim))
    cluster = rng.permutation(np.arange(n_items) % n_clusters)
    features = centers[cluster] + noise * rng.standard_normal((n_items, dim))
    rank = rng.permutation(n_items) + 1
    weight = rank.astype(np.float64) ** -zipf_s
    return Catalog(
        features=features.astype(np.float32),
        cluster=cluster.astype(np.int64),
        popularity=weight / weight.sum(),
    )


def sample_pairs(rng: np.random.Generator, catalog: Catalog, n_pairs: int) -> Pairs:
    anchor = rng.choice(catalog.n_items, size=n_pairs, p=catalog.popularity)
    positive = np.empty(n_pairs, dtype=np.int64)
    anchor_cluster = catalog.cluster[anchor]
    for c in np.unique(anchor_cluster):
        members = np.flatnonzero(catalog.cluster == c)
        weight = catalog.popularity[members]
        rows = np.flatnonzero(anchor_cluster == c)
        draw = rng.choice(members, size=rows.size, p=weight / weight.sum())
        clash = draw == anchor[rows]
        while clash.any():
            draw[clash] = rng.choice(members, size=int(clash.sum()), p=weight / weight.sum())
            clash = draw == anchor[rows]
        positive[rows] = draw
    return Pairs(anchor=anchor.astype(np.int64), positive=positive)


def mutual_information(catalog: Catalog) -> float:
    marginal = np.zeros(catalog.n_items)
    blocks = []
    for c in np.unique(catalog.cluster):
        members = np.flatnonzero(catalog.cluster == c)
        w = catalog.popularity[members]
        cond = w[None, :] / (w.sum() - w[:, None])
        np.fill_diagonal(cond, 0.0)
        marginal[members] += (w[:, None] * cond).sum(axis=0)
        blocks.append((members, w, cond))
    total = 0.0
    for members, w, cond in blocks:
        ratio = np.where(cond > 0, cond / marginal[members][None, :], 1.0)
        total += float((w[:, None] * cond * np.log(ratio)).sum())
    return total


def count_overlapping_rows(positive: np.ndarray) -> int:
    _, inverse, counts = np.unique(positive, return_inverse=True, return_counts=True)
    return int((counts[inverse] > 1).sum())
