from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
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


def build_hitl_agent(
    provider: BaseChatModel,
    system_prompt: str,
    tools: Sequence[BaseTool],
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> Runnable[Any, Any]:
    """Build the same ReAct agent, but wired for Human-in-the-Loop approval.

    Two additions over build_agent:

    - A checkpointer (MemorySaver by default) so the graph can pause mid-run and
      be resumed later against the same thread_id.
    - The expectation that one or more tools call langgraph.types.interrupt()
      before a destructive action (see cli.tools.write_file). When a tool
      interrupts, agent.invoke / agent.stream surfaces an "__interrupt__" payload
      instead of finishing; the caller approves or denies with
      Command(resume=...) and re-invokes against the same config.

    The interrupt lives inside the tool rather than via interrupt_before so that
    the human sees the concrete action (path, byte count, preview) rather than a
    bare "about to call write_file" signal.
    """

    if not tools:
        raise ValueError("tools must be non-empty for an agent to be useful.")

    return create_react_agent(
        model=provider,
        tools=list(tools),
        prompt=system_prompt,
        checkpointer=checkpointer or MemorySaver(),
    )
