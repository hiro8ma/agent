"""Deterministic offline embeddings + top-k cosine search for the semantic strategy.

faiss is intentionally not a dependency: with only six tools a numpy cosine scan is
exact and removes a heavy native wheel from the offline path. The same code path runs
faiss if it is importable, but never requires it.

When an embeddings key is present we use the real provider (via the core factory).
Without one we fall back to a character tri-gram hashing embedding so top-k selection
stays deterministic and runs with zero credentials. The hashing vector is a bag of
hashed n-grams: it captures lexical overlap, which is enough to route the demo queries
to the right tool, but it is *not* semantic — that degradation is the WHY for using a
real embedding model in production.
"""

from __future__ import annotations

import hashlib
import math
import os

_DIM = 256


def _ngrams(text: str, n: int = 3) -> list[str]:
    t = f" {text.lower().strip()} "
    return [t[i : i + n] for i in range(max(len(t) - n + 1, 0))] or [t]


def fake_embed(text: str) -> list[float]:
    """Hashing tri-gram embedding. Deterministic, key-free, lexical-overlap only."""

    vec = [0.0] * _DIM
    for gram in _ngrams(text):
        h = int(hashlib.md5(gram.encode()).hexdigest(), 16)  # noqa: S324  (not security)
        vec[h % _DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class Embedder:
    """Embeds text, preferring a real provider, falling back to fake hashing."""

    def __init__(self, use_fake: bool) -> None:
        self.is_fake = use_fake or not os.environ.get("OPENAI_API_KEY")
        self._provider = None
        if not self.is_fake:
            from core.providers.factory import select_embeddings

            self._provider = select_embeddings()

    def embed(self, text: str) -> list[float]:
        if self._provider is not None:
            return list(self._provider.embed_query(text))
        return fake_embed(text)

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if self._provider is not None:
            return [list(v) for v in self._provider.embed_documents(texts)]
        return [fake_embed(t) for t in texts]
