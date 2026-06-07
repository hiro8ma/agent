"""Real Elasticsearch / Qdrant connectors, loaded only when HELPDESK_BACKEND=real.

These are intentionally lazy: importing `tools.py` never pulls in the clients, so the
fake-backed graph runs with no extra dependencies. All connection settings come from
environment variables — never hardcode hosts, indices, or credentials.

Required env (real mode):
    ES_URL, ES_INDEX                       — Elasticsearch keyword backend
    QDRANT_URL, QDRANT_COLLECTION          — Qdrant vector backend
    QDRANT_API_KEY                         — optional, for managed Qdrant
"""

from __future__ import annotations

import os

from .tools import Document, Retriever


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"HELPDESK_BACKEND=real requires the {name} environment variable.")
    return value


class ElasticsearchRetriever:
    """BM25 keyword search against an Elasticsearch index."""

    def __init__(self, url: str, index: str) -> None:
        from elasticsearch import Elasticsearch  # lazy: only needed in real mode

        self._client = Elasticsearch(url)
        self._index = index

    def search(self, query: str, top_k: int) -> list[Document]:
        resp = self._client.search(
            index=self._index,
            query={"multi_match": {"query": query, "fields": ["title", "body"]}},
            size=top_k,
        )
        hits = resp.get("hits", {}).get("hits", [])
        return [
            Document(
                source=str(hit.get("_id", "manual")),
                text=str(hit.get("_source", {}).get("body", "")),
                score=float(hit.get("_score", 0.0)),
            )
            for hit in hits
        ]


class QdrantRetriever:
    """Cosine-similarity vector search against a Qdrant collection.

    Embeds the query with fastembed (already a project dependency) so the connector is
    self-contained and does not assume a server-side inference pipeline.
    """

    def __init__(self, url: str, collection: str, api_key: str | None) -> None:
        from fastembed import TextEmbedding  # lazy
        from qdrant_client import QdrantClient  # lazy

        self._client = QdrantClient(url=url, api_key=api_key)
        self._collection = collection
        self._embedder = TextEmbedding()

    def search(self, query: str, top_k: int) -> list[Document]:
        vector = next(iter(self._embedder.embed([query]))).tolist()
        results = self._client.search(
            collection_name=self._collection,
            query_vector=vector,
            limit=top_k,
        )
        return [
            Document(
                source=str(point.id),
                text=str((point.payload or {}).get("text", "")),
                score=float(point.score),
            )
            for point in results
        ]


def build_elasticsearch_retriever() -> Retriever:
    return ElasticsearchRetriever(_require("ES_URL"), _require("ES_INDEX"))


def build_qdrant_retriever() -> Retriever:
    return QdrantRetriever(
        _require("QDRANT_URL"),
        _require("QDRANT_COLLECTION"),
        os.environ.get("QDRANT_API_KEY"),
    )
