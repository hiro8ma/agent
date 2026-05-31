from __future__ import annotations

from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from cli.retriever import format_documents


class SearchDocumentsInput(BaseModel):
    """Input schema for the search_documents tool."""

    query: str = Field(
        ...,
        description=(
            "Focused natural-language query used to retrieve passages from the indexed PDF."
        ),
    )


def build_search_documents_tool(
    retriever: BaseRetriever,
    name: str = "search_documents",
    description: str = (
        "Search the indexed PDF for passages relevant to a query. "
        "Use this whenever the user question requires knowledge from the document. "
        "Input is a single natural-language query string. "
        "Output is the top matching chunks with page metadata when available."
    ),
) -> BaseTool:
    """Wrap a retriever as a StructuredTool with a pydantic input schema.

    Preferred over the schema-less cli.retriever.wrap_as_tool: this version
    advertises a typed args_schema (SearchDocumentsInput) so the LLM emits
    structured tool calls with a named, validated query field.
    """

    def _run(query: str) -> str:
        docs = retriever.invoke(query)
        return format_documents(docs)

    return StructuredTool.from_function(
        name=name,
        description=description,
        func=_run,
        args_schema=SearchDocumentsInput,
    )
