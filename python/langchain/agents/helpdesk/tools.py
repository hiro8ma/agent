"""Hybrid RAG retrieval tools for the helpdesk agent.

Two retrievers cover complementary query shapes:

- `search_manual` — keyword search (Elasticsearch-style). Best for exact matches on
  product numbers, error codes, and domain jargon found in manuals / release notes.
- `search_qa` — vector search (Qdrant-style). Best for semantic similarity against
  past question / answer pairs where the wording differs but the intent matches.

Both default to an in-memory fake backend so the graph runs with zero infrastructure.
Set `HELPDESK_BACKEND=real` to swap in the Elasticsearch / Qdrant connectors instead;
those are imported lazily so the import of this module never requires the clients.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Protocol

from langchain_core.tools import StructuredTool


@dataclass(frozen=True)
class Document:
    """A single retrieved passage with its source label and relevance score."""

    source: str
    text: str
    score: float


class Retriever(Protocol):
    """Common shape for keyword and vector backends, real or fake."""

    def search(self, query: str, top_k: int) -> list[Document]: ...


# --- Sample corpus (generic, fictional system "XYZ"; no real product data) ---

_MANUAL_CORPUS: list[tuple[str, str]] = [
    ("manual:XYZ-1001", "Error code E-204 on system XYZ means the session token expired. "
     "Re-authenticate from the admin console to refresh the token."),
    ("manual:XYZ-1002", "Release note for XYZ v3.2: the export endpoint now requires the "
     "scope `data:export`. Requests without it return HTTP 403."),
    ("manual:XYZ-1003", "To rotate an API key for system XYZ, open Settings > Credentials "
     "and select Rotate. The previous key stays valid for 24 hours."),
    ("manual:XYZ-1004", "Error code E-500 indicates an internal batch job failure. Check the "
     "job queue dashboard and retry the failed job."),
]

_QA_CORPUS: list[tuple[str, str]] = [
    ("qa:0001", "Q: My login keeps getting kicked out. A: Your token likely expired; "
     "sign in again and the session will be restored."),
    ("qa:0002", "Q: Downloads are failing with a permission error. A: Ask an admin to grant "
     "the export permission to your account."),
    ("qa:0003", "Q: How do I change my access key safely? A: Use the rotate option in "
     "settings; the old key works for a day so nothing breaks."),
    ("qa:0004", "Q: The nightly process didn't run. A: A background job failed; rerun it "
     "from the dashboard or contact support if it repeats."),
]


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9-]+", text.lower()))


class FakeKeywordRetriever:
    """Token-overlap keyword search. Stand-in for an Elasticsearch BM25 query."""

    def __init__(self, corpus: list[tuple[str, str]]) -> None:
        self._corpus = corpus

    def search(self, query: str, top_k: int) -> list[Document]:
        q_tokens = _tokenize(query)
        scored: list[Document] = []
        for source, text in self._corpus:
            overlap = len(q_tokens & _tokenize(text))
            if overlap:
                scored.append(Document(source=source, text=text, score=float(overlap)))
        scored.sort(key=lambda d: d.score, reverse=True)
        return scored[:top_k]


class FakeVectorRetriever:
    """Jaccard-similarity search. Stand-in for a Qdrant cosine-similarity query.

    Jaccard over token sets is a deterministic, dependency-free proxy for semantic
    similarity, enough to demonstrate vector-style retrieval offline.
    """

    def __init__(self, corpus: list[tuple[str, str]]) -> None:
        self._corpus = corpus

    def search(self, query: str, top_k: int) -> list[Document]:
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []
        scored: list[Document] = []
        for source, text in self._corpus:
            d_tokens = _tokenize(text)
            union = q_tokens | d_tokens
            if not union:
                continue
            similarity = len(q_tokens & d_tokens) / len(union)
            if similarity > 0:
                scored.append(Document(source=source, text=text, score=round(similarity, 3)))
        scored.sort(key=lambda d: d.score, reverse=True)
        return scored[:top_k]


def _build_keyword_retriever() -> Retriever:
    if os.environ.get("HELPDESK_BACKEND") == "real":
        from .connectors import build_elasticsearch_retriever

        return build_elasticsearch_retriever()
    return FakeKeywordRetriever(_MANUAL_CORPUS)


def _build_vector_retriever() -> Retriever:
    if os.environ.get("HELPDESK_BACKEND") == "real":
        from .connectors import build_qdrant_retriever

        return build_qdrant_retriever()
    return FakeVectorRetriever(_QA_CORPUS)


def _format(docs: list[Document]) -> str:
    if not docs:
        return "(no matching documents)"
    return "\n".join(f"[{d.source} score={d.score}] {d.text}" for d in docs)


def search_manual(query: str, top_k: int = 3) -> str:
    """Keyword search over product manuals and release notes (Elasticsearch-style).

    Use this for exact-match lookups: error codes (e.g. E-204), product numbers,
    version strings, API scope names, or other precise domain terms.
    """

    return _format(_build_keyword_retriever().search(query, top_k))


def search_qa(query: str, top_k: int = 3) -> str:
    """Vector search over past question / answer pairs (Qdrant-style).

    Use this for semantic lookups where the user describes a problem in their own
    words and you want similar past cases rather than an exact term match.
    """

    return _format(_build_vector_retriever().search(query, top_k))


SEARCH_MANUAL_TOOL = StructuredTool.from_function(
    func=search_manual,
    name="search_manual",
    description=(
        "Keyword search over manuals and release notes. Prefer for exact terms: "
        "error codes, product numbers, versions, API scopes."
    ),
)

SEARCH_QA_TOOL = StructuredTool.from_function(
    func=search_qa,
    name="search_qa",
    description=(
        "Vector search over past Q&A. Prefer for paraphrased, symptom-style questions "
        "where semantic similarity beats exact keyword match."
    ),
)

ALL_TOOLS = [SEARCH_MANUAL_TOOL, SEARCH_QA_TOOL]
