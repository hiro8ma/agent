"""Searching a memory store: lexical BM25 vs semantic embeddings, and why hybrid wins.

Knowledge vs memory (the distinction this module operates under):

- *Knowledge* (RAG) is static external reference material — docs, manuals, a wiki —
  indexed once and queried read-only. It does not grow because of a conversation.
- *Memory* is the agent's own past — notes it took, things a user told it, prior
  turns — written as the agent runs and read back later. This module retrieves memory.

Two ways to retrieve, with opposite failure modes:

- BM25 (lexical): scores documents by exact term overlap, weighting rare terms and
  damping document length. Unbeatable for *proper nouns / IDs / exact tokens* a user
  said, because those terms appear verbatim. Blind to paraphrase: query "stipend" will
  not match a memory that only ever says "allowance".
- Semantic (embeddings + cosine): matches meaning, so paraphrase and concept queries
  hit even with zero shared words. Weaker on rare exact tokens, where an embedding can
  smear a unique ID into nearby-but-wrong neighbours.

Production memory stores run both and fuse the rankings (hybrid retrieval — e.g.
reciprocal-rank fusion), so an exact-token query and a concept query are both served.
`hybrid_search` below is the minimal version of that idea.

BM25 is implemented in pure Python on purpose: it keeps the offline path dependency-free
(no `rank_bm25` wheel). The semantic side reuses the deterministic offline embedder from
`agents.tool_selection.embed`, so this whole module runs with zero credentials.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from agents.tool_selection.embed import Embedder, cosine

_TOKEN = re.compile(r"[a-z0-9]+")

# Generic, fictional "agent memory" corpus: notes the agent wrote during past sessions.
DEFAULT_CORPUS: list[str] = [
    "User Zephyr asked to schedule the quarterly planning sync for next Tuesday.",
    "The team agreed to pay each intern a monthly allowance of 500 credits.",
    "Project Halcyon's deploy was blocked by a flaky integration test on the gateway.",
    "User said they prefer dark mode and concise replies with no preamble.",
    "We rotated the staging database password and stored it in the vault.",
    "The onboarding guide should explain how new hires get reimbursed for expenses.",
    "Customer Vega reported the export button does nothing on large reports.",
    "Decided to migrate the cron jobs to the new scheduler before the audit.",
]


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass
class MemoryStore:
    """An in-memory corpus indexed for both BM25 and semantic search."""

    documents: list[str]
    _doc_tokens: list[list[str]] = field(default_factory=list, repr=False)
    _doc_freq: dict[str, int] = field(default_factory=dict, repr=False)
    _avg_len: float = 0.0
    _embedder: Embedder | None = field(default=None, repr=False)
    _doc_vectors: list[list[float]] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self._doc_tokens = [tokenize(d) for d in self.documents]
        df: dict[str, int] = defaultdict(int)
        for tokens in self._doc_tokens:
            for term in set(tokens):
                df[term] += 1
        self._doc_freq = dict(df)
        total = sum(len(t) for t in self._doc_tokens)
        self._avg_len = total / len(self._doc_tokens) if self._doc_tokens else 0.0

    def _ensure_vectors(self, use_fake: bool) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder(use_fake=use_fake)
            self._doc_vectors = self._embedder.embed_many(self.documents)
        return self._embedder


def _idf(store: MemoryStore, term: str) -> float:
    n = len(store.documents)
    df = store._doc_freq.get(term, 0)
    # BM25 idf: rare terms (low df) score high, which is why exact proper nouns win.
    return math.log(1.0 + (n - df + 0.5) / (df + 0.5))


def bm25_search(
    store: MemoryStore,
    query: str,
    top_k: int = 3,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[tuple[int, float]]:
    """Rank documents by BM25. Returns (doc_index, score) sorted high-to-low."""

    q_terms = tokenize(query)
    scored: list[tuple[int, float]] = []
    for i, tokens in enumerate(store._doc_tokens):
        counts = Counter(tokens)
        dl = len(tokens)
        score = 0.0
        for term in q_terms:
            tf = counts.get(term, 0)
            if tf == 0:
                continue
            denom = tf + k1 * (1.0 - b + b * dl / (store._avg_len or 1.0))
            score += _idf(store, term) * (tf * (k1 + 1.0)) / denom
        if score > 0.0:
            scored.append((i, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def semantic_search(
    store: MemoryStore,
    query: str,
    top_k: int = 3,
    use_fake: bool = False,
) -> list[tuple[int, float]]:
    """Rank documents by cosine similarity of embeddings. Returns (doc_index, score)."""

    embedder = store._ensure_vectors(use_fake)
    q_vec = embedder.embed(query)
    scored = [(i, cosine(q_vec, vec)) for i, vec in enumerate(store._doc_vectors)]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def hybrid_search(
    store: MemoryStore,
    query: str,
    top_k: int = 3,
    use_fake: bool = False,
) -> list[tuple[int, float]]:
    """Fuse BM25 and semantic rankings with reciprocal-rank fusion.

    RRF avoids comparing incompatible score scales: it sums 1/(rank+const) across both
    rankers, so a doc ranked high by either method floats up. This is the minimal form
    of what a production hybrid memory store does.
    """

    const = 60.0
    fused: dict[int, float] = defaultdict(float)
    for ranking in (
        bm25_search(store, query, top_k=len(store.documents)),
        semantic_search(store, query, top_k=len(store.documents), use_fake=use_fake),
    ):
        for rank, (idx, _score) in enumerate(ranking):
            fused[idx] += 1.0 / (const + rank)
    merged = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    return merged[:top_k]
