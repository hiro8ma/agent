"""Run helpers for the supervisor multi-agent graph (streaming + final answer)."""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

from langchain_core.language_models import BaseChatModel

from .graph import DEFAULT_TOOL_ROUNDS, build_graph, initial_state


def _select_model(use_fake: bool) -> tuple[BaseChatModel, bool]:
    """Return (model, is_fake).

    Falls back to the offline FakeChatModel when no LLM key is set, so routing and the
    specialist tool loop stay runnable without credentials.
    """

    if not use_fake and os.environ.get("OPENAI_API_KEY"):
        from core.providers.factory import select_provider

        return select_provider(), False

    from .fake import FakeChatModel

    return FakeChatModel(), True


def stream(
    query: str,
    max_tool_rounds: int = DEFAULT_TOOL_ROUNDS,
    use_fake: bool = False,
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield (node_name, state_update) for each step as the graph runs."""

    model, _ = _select_model(use_fake)
    app = build_graph(model)
    for chunk in app.stream(initial_state(query, max_tool_rounds), {"recursion_limit": 50}):
        yield from chunk.items()


def run(
    query: str,
    max_tool_rounds: int = DEFAULT_TOOL_ROUNDS,
    use_fake: bool = False,
) -> dict[str, Any]:
    """Run the graph to completion and return the final state."""

    model, _ = _select_model(use_fake)
    app = build_graph(model)
    return dict(app.invoke(initial_state(query, max_tool_rounds), {"recursion_limit": 50}))
