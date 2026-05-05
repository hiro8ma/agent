from __future__ import annotations

from collections.abc import Sequence

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever

DEFAULT_COLLECTION = "py-phase2-collection"
DEFAULT_RETRIEVER_K = 4


def build_vector_store(
    documents: Sequence[Document],
    embeddings: Embeddings,
    collection_name: str = DEFAULT_COLLECTION,
    persist_directory: str | None = None,
) -> VectorStore:
    """Embed documents and load them into a Chroma vector store.

    py-phase2 stays in-memory by default (persist_directory=None). Pass a
    persist_directory to write the collection to disk for reuse across runs.
    """

    if not documents:
        raise ValueError("documents must be non-empty to build a vector store.")

    return Chroma.from_documents(
        documents=list(documents),
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )


def as_retriever(
    store: VectorStore,
    k: int = DEFAULT_RETRIEVER_K,
) -> VectorStoreRetriever:
    """Return a similarity-search retriever bound to the store."""

    if k <= 0:
        raise ValueError(f"k must be positive: {k}")
    return store.as_retriever(search_kwargs={"k": k})
