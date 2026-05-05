from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage

from cli.agent import build_agent
from cli.document_loader import load_pdf
from cli.text_splitter import split_documents
from cli.tools.search_documents import build_search_documents_tool
from cli.vector_store import as_retriever, build_vector_store
from core.providers.factory import select_embeddings, select_provider

from .prompt import SEARCH_AGENT_SYSTEM_PROMPT

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_RETRIEVER_K = 4


def run(
    pdf_path: str,
    question: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    retriever_k: int = DEFAULT_RETRIEVER_K,
) -> str:
    """Answer a question by indexing the PDF into Chroma and letting the agent search.

    Pipeline: load PDF -> split -> embed -> Chroma -> retriever -> wrap as Tool ->
    LangGraph create_react_agent invokes the Tool as needed.
    """

    docs = load_pdf(pdf_path)
    chunks = split_documents(docs, chunk_size, chunk_overlap)
    if not chunks:
        raise ValueError("PDF produced zero chunks; nothing to index.")

    embeddings = select_embeddings()
    store = build_vector_store(chunks, embeddings)
    retriever = as_retriever(store, k=retriever_k)
    search_tool = build_search_documents_tool(retriever)

    agent = build_agent(
        provider=select_provider(),
        system_prompt=SEARCH_AGENT_SYSTEM_PROMPT,
        tools=[search_tool],
    )

    result: dict[str, Any] = agent.invoke(
        {"messages": [{"role": "user", "content": question}]}
    )
    return _extract_final_answer(result)


def _extract_final_answer(result: dict[str, Any]) -> str:
    """Pull the last assistant message content out of the agent state."""

    messages: list[BaseMessage] = result.get("messages", [])
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            content = message.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: list[str] = []
                for part in content:
                    if isinstance(part, str):
                        parts.append(part)
                    elif isinstance(part, dict) and isinstance(part.get("text"), str):
                        parts.append(part["text"])
                if parts:
                    return "\n".join(parts)
    return "(agent produced no assistant message)"
