from __future__ import annotations

from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool, Tool


def format_documents(docs: Sequence[Document]) -> str:
    """Render retrieved Documents into a numbered, citation-friendly string."""

    if not docs:
        return "(no relevant context found)"

    blocks: list[str] = []
    for i, doc in enumerate(docs, start=1):
        page = doc.metadata.get("page")
        source = doc.metadata.get("source")
        header_parts = [f"chunk {i}"]
        if page is not None:
            header_parts.append(f"p. {page}")
        if source:
            header_parts.append(str(source))
        header = "[" + " | ".join(header_parts) + "]"
        blocks.append(f"{header}\n{doc.page_content.strip()}")
    return "\n\n".join(blocks)


def wrap_as_tool(
    retriever: BaseRetriever,
    name: str = "search_documents",
    description: str = (
        "Search the indexed PDF for passages relevant to a query. "
        "Input must be a focused natural-language query string. "
        "Returns the top matching chunks with page metadata when available."
    ),
) -> BaseTool:
    """Wrap a retriever as a single-input LangChain Tool returning a formatted string.

    This is the schema-less variant: Tool advertises only one freeform string
    argument, so the LLM cannot rely on a named, typed field. Prefer
    cli.tools.search_documents.build_search_documents_tool, which exposes a typed
    pydantic args_schema for structured tool calls. Both share the name
    "search_documents"; never register both in one agent.
    """

    def _run(query: str) -> str:
        docs = retriever.invoke(query)
        return format_documents(docs)

    return Tool(
        name=name,
        description=description,
        func=_run,
    )
