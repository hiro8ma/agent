from __future__ import annotations

import json
import os
from typing import Any

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

DEFAULT_MAX_RESULTS = 5


class WebSearchInput(BaseModel):
    """Input schema for the web_search tool."""

    query: str = Field(
        ...,
        description=(
            "Focused natural-language web search query. "
            "Prefer specific phrasing over broad keywords."
        ),
    )


def build_web_search_tool(
    max_results: int = DEFAULT_MAX_RESULTS,
    name: str = "web_search",
    description: str = (
        "Search the public web via Tavily and return ranked results with titles, "
        "URLs, and content snippets. Use this to gather facts before writing a report. "
        "Input is a single natural-language query string."
    ),
) -> BaseTool:
    """Wrap Tavily web search as a StructuredTool with a typed input schema.

    Reads TAVILY_API_KEY from the environment. Raises RuntimeError when the key
    is missing so misconfiguration surfaces before the first network call, matching
    how core.providers.openai fails on a missing OPENAI_API_KEY.
    """

    if not os.environ.get("TAVILY_API_KEY"):
        raise RuntimeError(
            "TAVILY_API_KEY is not set. Export it before running the web researcher."
        )

    backend = TavilySearchResults(max_results=max_results)

    def _run(query: str) -> str:
        results: Any = backend.invoke(query)
        return _format_results(results)

    return StructuredTool.from_function(
        name=name,
        description=description,
        func=_run,
        args_schema=WebSearchInput,
    )


def _format_results(results: Any) -> str:
    """Render Tavily results as a numbered, citation-friendly block."""

    if isinstance(results, str):
        return results
    if not isinstance(results, list) or not results:
        return "(web_search returned no results)"

    blocks: list[str] = []
    for i, item in enumerate(results, start=1):
        if not isinstance(item, dict):
            blocks.append(f"[{i}] {item}")
            continue
        url = item.get("url", "")
        title = item.get("title") or url or "(untitled)"
        content = (item.get("content") or "").strip()
        header = f"[{i}] {title}" + (f" — {url}" if url else "")
        blocks.append(f"{header}\n{content}" if content else header)
    return "\n\n".join(blocks)


def serialize_results(results: Any) -> str:
    """Best-effort JSON serialization, kept for callers that want raw payloads."""

    try:
        return json.dumps(results, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return str(results)
