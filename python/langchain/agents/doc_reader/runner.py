from __future__ import annotations

from collections.abc import Sequence

from langchain_core.documents import Document

from cli.chain import build_chain
from cli.document_loader import load_pdf
from cli.parsers import string_parser
from cli.text_splitter import split_documents
from core.providers.factory import select_provider
from core.types import RunInput, RunOutput

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_MAX_CHUNKS = 20


def run(
    pdf_path: str,
    question: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
) -> str:
    """Answer the question from the PDF.

    py-phase1 uses a 'stuff' strategy: load the PDF, split into chunks, and
    feed the leading max_chunks into the prompt as context. Real RAG
    (embedding + vector store + retrieval) lands in py-phase2.
    """

    payload = RunInput(
        pdf_path=pdf_path,
        question=question,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_chunks=max_chunks,
    )

    docs = load_pdf(payload.pdf_path)
    chunks = split_documents(docs, payload.chunk_size, payload.chunk_overlap)
    selected = chunks[: payload.max_chunks]
    context = _format_context(selected)

    from .prompt import DOC_READER_SYSTEM_PROMPT

    chain = build_chain(
        provider=select_provider(),
        system_prompt=DOC_READER_SYSTEM_PROMPT,
        parser=string_parser(),
    )

    answer = chain.invoke(
        {
            "question": payload.question,
            "context": context,
            "chat_history": [],
        }
    )

    if not isinstance(answer, str):
        answer = str(answer)

    result = RunOutput(
        answer=answer,
        used_chunks=len(selected),
        total_chunks=len(chunks),
    )
    return result.answer


def _format_context(chunks: Sequence[Document]) -> str:
    """Render chunks as a numbered list with page metadata when available."""

    if not chunks:
        return "(no context extracted)"

    blocks: list[str] = []
    for i, doc in enumerate(chunks, start=1):
        page = doc.metadata.get("page")
        header = f"[chunk {i}]" if page is None else f"[chunk {i} | p. {page}]"
        blocks.append(f"{header}\n{doc.page_content.strip()}")
    return "\n\n".join(blocks)
