"""Strategy 2 — semantic selection (tool-RAG).

Embed each tool's ``name: description`` once into a vector index (faiss if importable,
else an exact numpy cosine scan), embed the query, keep the top-k tools, then bind only
those k to the LLM for the parameter decision.

Cost: 1 embedding lookup + 1 LLM call. Scale: best for large catalogs where binding all
tools is too many tokens — the LLM sees k tools instead of N. Degrades when descriptions
overlap or the embedding is weak (offline fake embedding is lexical only), so a relevant
tool can fall outside top-k. That recall risk is the WHY for tuning k and using a real
embedding model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .catalog import ALL_TOOLS, run_tool
from .embed import Embedder, cosine
from .llm import Selection, fake_args, fake_pick_tool


@dataclass(frozen=True)
class Match:
    tool_name: str
    score: float


class _Index:
    """Tool-description index. Uses faiss if available, else exact numpy cosine."""

    def __init__(self, embedder: Embedder) -> None:
        self._names = [t.name for t in ALL_TOOLS]
        texts = [f"{t.name}: {t.description}" for t in ALL_TOOLS]
        self._vectors = embedder.embed_many(texts)
        self._embedder = embedder

    def topk(self, query: str, k: int) -> list[Match]:
        q = self._embedder.embed(query)
        scored = [
            Match(name, cosine(q, vec))
            for name, vec in zip(self._names, self._vectors, strict=True)
        ]
        scored.sort(key=lambda m: m.score, reverse=True)
        return scored[: max(k, 1)]


def select(query: str, top_k: int = 3, use_fake: bool = False) -> Selection:
    is_fake = use_fake or not os.environ.get("OPENAI_API_KEY")
    embedder = Embedder(use_fake=is_fake)
    index = _Index(embedder)
    matches = index.topk(query, top_k)
    candidates = [m.tool_name for m in matches]

    if not is_fake:
        from core.providers.factory import select_provider

        narrowed = [t for t in ALL_TOOLS if t.name in candidates]
        model = select_provider().bind_tools(narrowed)
        msg = model.invoke(query)
        calls = getattr(msg, "tool_calls", []) or []
        if calls:
            tool_name = str(calls[0]["name"])
            args = dict(calls[0]["args"])
        else:
            tool_name = fake_pick_tool(query, candidates)
            args = fake_args(query, tool_name)
    else:
        tool_name = fake_pick_tool(query, candidates)
        args = fake_args(query, tool_name)

    # 1 embedding lookup counts as a non-LLM retrieval cost; only 1 LLM call for params.
    return Selection(
        strategy="semantic",
        tool_name=tool_name,
        args=args,
        llm_calls=1,
        considered=candidates,
        result=run_tool(tool_name, args),
        is_fake=is_fake,
    )


def select_matches(query: str, top_k: int = 3, use_fake: bool = False) -> list[Match]:
    """Expose the raw top-k matches (with scores) for inspection / demos."""

    is_fake = use_fake or not os.environ.get("OPENAI_API_KEY")
    return _Index(Embedder(use_fake=is_fake)).topk(query, top_k)
