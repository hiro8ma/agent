"""Semantic tool selection (tool-RAG) for MCP tools.

Standard MCP clients bind *every* tool the servers expose to the LLM. As the
tool catalog grows that wastes context and dilutes tool-choice accuracy. tool-RAG
instead embeds each tool's ``name: description`` once, embeds the user query, and
binds only the top-k most similar tools. This module is the retrieval half; the
binding half stays in ``runner.run`` / the demo.

Embeddings: a key-free local FastEmbed (ONNX) model by default, with OpenAI as a
fallback when EMBEDDING_PROVIDER=openai and OPENAI_API_KEY is set. Similarity is
plain cosine over numpy arrays, so no vector DB dependency is required (the
FAISS example from the book reduces to this for a handful of tools).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.tools import BaseTool

from core.providers.embeddings import (
    build_fastembed_embeddings,
    build_openai_embeddings,
)


def _select_tool_embeddings() -> Embeddings:
    """Pick an embeddings backend for tool selection (local-first, key-free).

    Default is FastEmbed so the path works with no API key. Set
    EMBEDDING_PROVIDER=openai (with OPENAI_API_KEY) to use OpenAI embeddings.
    """

    provider = (os.environ.get("EMBEDDING_PROVIDER") or "fastembed").lower()
    if provider == "openai":
        return build_openai_embeddings()
    return build_fastembed_embeddings()


def tool_document(tool: BaseTool) -> str:
    """Render a tool as the text that gets embedded: ``name: description``."""

    description = (tool.description or "").strip()
    return f"{tool.name}: {description}" if description else tool.name


def _cosine_top_k(
    query_vec: np.ndarray, doc_vecs: np.ndarray, k: int
) -> list[tuple[int, float]]:
    """Return (index, score) for the top-k rows of doc_vecs by cosine similarity."""

    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    doc_norms = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-12)
    scores = doc_norms @ query_norm
    order = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in order]


@dataclass
class ToolMatch:
    """One scored tool from semantic selection."""

    tool: BaseTool
    score: float


def select_tools_semantically(
    query: str,
    tools: list[BaseTool],
    k: int = 3,
    embeddings: Embeddings | None = None,
) -> list[ToolMatch]:
    """Embed each tool's ``name: description``, embed the query, return top-k.

    This is the retrieval step of tool-RAG: instead of binding all ``len(tools)``
    tools, the caller binds only the returned ``min(k, len(tools))`` tools. Order
    is by descending cosine similarity.
    """

    if not tools:
        return []
    k = max(1, min(k, len(tools)))

    embedder = embeddings or _select_tool_embeddings()
    docs = [tool_document(t) for t in tools]
    doc_vecs = np.asarray(embedder.embed_documents(docs), dtype=np.float32)
    query_vec = np.asarray(embedder.embed_query(query), dtype=np.float32)

    return [
        ToolMatch(tool=tools[i], score=score)
        for i, score in _cosine_top_k(query_vec, doc_vecs, k)
    ]
