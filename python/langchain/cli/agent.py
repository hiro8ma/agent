from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent


def build_agent(
    provider: BaseChatModel,
    system_prompt: str,
    tools: Sequence[BaseTool],
) -> Runnable[Any, Any]:
    """Construct a tool-calling agent with LangGraph's prebuilt ReAct loop.

    Adopts langgraph.prebuilt.create_react_agent (LangChain v1 / LangGraph 1.0
    stable path). The newer langchain.agents.create_agent shim wraps the same
    primitive; we go straight to the source so behavior is unambiguous.
    """

    if not tools:
        raise ValueError("tools must be non-empty for an agent to be useful.")

    return create_react_agent(
        model=provider,
        tools=list(tools),
        prompt=system_prompt,
    )
