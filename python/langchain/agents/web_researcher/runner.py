from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Command

from cli.agent import build_hitl_agent
from cli.tools.web_search import build_web_search_tool
from cli.tools.write_file import build_write_file_tool
from core.providers.factory import select_provider

from .prompt import WEB_RESEARCHER_SYSTEM_PROMPT

DEFAULT_WORKSPACE = "workspace"
DEFAULT_MAX_RESULTS = 5

# Approval callback: receives the interrupt payload, returns True to approve.
ApprovalFn = Callable[[dict[str, Any]], bool]


def build_web_researcher(
    workspace: str | Path = DEFAULT_WORKSPACE,
    max_results: int = DEFAULT_MAX_RESULTS,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> Runnable[Any, Any]:
    """Compile the HITL web-research agent.

    Returned graph is shared by the CLI (terminal approval) and the Streamlit GUI
    (button approval). Both drive it through .stream/.invoke + Command(resume=...).
    Requires TAVILY_API_KEY (web_search) and an LLM key (select_provider).
    """

    web_search = build_web_search_tool(max_results=max_results)
    write_file = build_write_file_tool(workspace=workspace)

    return build_hitl_agent(
        provider=select_provider(),
        system_prompt=WEB_RESEARCHER_SYSTEM_PROMPT,
        tools=[web_search, write_file],
        checkpointer=checkpointer,
    )


def run(
    topic: str,
    approve: ApprovalFn,
    workspace: str | Path = DEFAULT_WORKSPACE,
    max_results: int = DEFAULT_MAX_RESULTS,
    thread_id: str = "web-researcher",
) -> str:
    """Research a topic and (after approval) save an HTML report.

    The agent pauses before every write_file. The `approve` callback decides
    whether each write proceeds. The CLI passes a stdin prompt; the GUI passes a
    button-driven decision. Loops until the graph runs to completion.
    """

    agent = build_web_researcher(workspace=workspace, max_results=max_results)
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    state: dict[str, Any] = agent.invoke(
        {"messages": [{"role": "user", "content": topic}]}, config
    )

    while "__interrupt__" in state:
        payload = state["__interrupt__"][0].value
        decision = "approve" if approve(payload) else "deny"
        state = agent.invoke(Command(resume=decision), config)

    return _extract_final_answer(state)


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
