from __future__ import annotations

from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    docs: Sequence[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
) -> list[Document]:
    """Split Documents into smaller chunks for prompt assembly."""

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive: {chunk_size}")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap must satisfy 0 <= overlap < chunk_size, got "
            f"overlap={chunk_overlap}, size={chunk_size}"
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(list(docs))
