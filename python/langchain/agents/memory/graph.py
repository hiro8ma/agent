"""A one-node chat graph that demonstrates session-spanning memory via a checkpointer.

Why a checkpointer *is* memory:

A graph node is a pure function of the state it is handed. On its own, each `invoke`
starts from a fresh state, so turn 2 never sees turn 1 — the model is amnesiac. A
checkpointer persists the state after every super-step, keyed by `thread_id`. The next
`invoke` with the same `thread_id` loads that saved state first, so the prior messages
are replayed into the node. That replay — not anything inside the model — is what makes
the agent remember. Swap `InMemorySaver` for a SQLite/Postgres saver and the same
mechanism spans process restarts and sessions; only the storage backend changes.

`InMemorySaver` keeps the deps offline (no extra wheel). The state uses the standard
`add_messages` reducer so each turn appends to history rather than overwriting it.
"""

from __future__ import annotations

import os
from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph


class ChatState(TypedDict):
    """Conversation state. `add_messages` appends each turn instead of replacing it."""

    messages: Annotated[list[BaseMessage], add_messages]


def _select_model(use_fake: bool) -> BaseChatModel:
    if not use_fake and os.environ.get("OPENAI_API_KEY"):
        from core.providers.factory import select_provider

        return select_provider()

    from .fake import FakeChatModel

    return FakeChatModel()


def build_chat_graph(
    model: BaseChatModel,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> CompiledStateGraph[Any, Any, Any]:
    """Compile the chat graph. Pass a checkpointer to enable cross-turn memory."""

    def chat(state: ChatState) -> dict[str, Any]:
        return {"messages": [model.invoke(state["messages"])]}

    workflow: StateGraph[Any, Any, Any, Any] = StateGraph(ChatState)
    workflow.add_node("chat", chat)
    workflow.add_edge(START, "chat")
    workflow.add_edge("chat", END)
    return workflow.compile(checkpointer=checkpointer)


def reply_without_memory(turns: list[str], use_fake: bool = False) -> list[str]:
    """Run each turn through a *checkpointer-less* graph — the amnesiac baseline.

    Every turn is a cold invoke: the node only ever sees the single current message,
    so an introduction in turn 1 is invisible to turn 2.
    """

    model = _select_model(use_fake)
    app = build_chat_graph(model, checkpointer=None)
    replies: list[str] = []
    for turn in turns:
        result = app.invoke({"messages": [HumanMessage(content=turn)]})
        replies.append(_last_text(result["messages"]))
    return replies


def run_session(turns: list[str], thread_id: str = "demo", use_fake: bool = False) -> list[str]:
    """Run each turn through a checkpointed graph under one `thread_id` — memory on.

    The saver reloads prior state before each turn, so earlier messages are replayed
    into the node and the model can answer about them.
    """

    model = _select_model(use_fake)
    app = build_chat_graph(model, checkpointer=InMemorySaver())
    config: Any = {"configurable": {"thread_id": thread_id}}
    replies: list[str] = []
    for turn in turns:
        result = app.invoke({"messages": [HumanMessage(content=turn)]}, config)
        replies.append(_last_text(result["messages"]))
    return replies


def _last_text(messages: list[BaseMessage]) -> str:
    content = messages[-1].content if messages else ""
    return content if isinstance(content, str) else str(content)
