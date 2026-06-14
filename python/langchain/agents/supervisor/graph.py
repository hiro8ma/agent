"""Supervisor-style multi-agent system on a LangGraph StateGraph.

Flow:

                ┌─ inventory ──┐
    supervisor ─┼─ transport ──┼─ END
                └─ supplier ───┘

`supervisor` reads the request and names one specialist. `route_to_specialist` turns
that name into the next node. Each specialist owns a disjoint tool group and runs its
own tool-calling loop (model → tool_calls → run tools → ToolMessage → model again) until
the model stops requesting tools, then writes the answer and goes to END.

Why specialize: a single agent holding all nine tools needs a prompt describing all of
them and picks from a nine-way menu every turn. Splitting into three specialists shrinks
each prompt and each tool menu to one domain, which cuts mis-selection. The price is the
supervisor's extra routing hop and the coordination state it carries — the classic
multi-agent trade-off. Move from single to multi-agent only once one of those costs (a
bloated prompt, frequent wrong-tool calls, mixed-domain context) actually shows up.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .prompt import SPECIALIST_SYSTEM_PROMPT, SUPERVISOR_SYSTEM_PROMPT
from .tools import SPECIALISTS, SPECIALISTS_BY_NAME, run_tool, tools_for

SPECIALIST_NAMES = tuple(s.name for s in SPECIALISTS)
DEFAULT_TOOL_ROUNDS = 4


class AgentState(TypedDict):
    """Shared state for the supervisor graph.

    Reducers are chosen per field on purpose:
    - `messages` uses `operator.add` so the supervisor decision, each specialist turn,
      and every ToolMessage append to one ordered transcript instead of overwriting it.
    - `operation` is plain (last-write-wins): it is a small control dict (the chosen
      route, the final answer, a round counter), so each node overwrites the keys it
      owns rather than accumulating stale copies.
    """

    operation: dict[str, Any]
    messages: Annotated[list[BaseMessage], operator.add]


def _text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content.strip()
    parts = [p for p in content if isinstance(p, str)]
    return "\n".join(parts).strip()


def _build_supervisor(model: BaseChatModel) -> Any:
    def supervisor(state: AgentState) -> dict[str, Any]:
        query = state["operation"]["query"]
        response = model.invoke(
            [SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT), HumanMessage(content=query)]
        )
        choice = _text(response).lower()
        # Default to the first specialist when the model returns anything off-menu.
        target = next((n for n in SPECIALIST_NAMES if n in choice), SPECIALIST_NAMES[0])
        return {
            "operation": {**state["operation"], "route": target},
            "messages": [AIMessage(content=f"[supervisor] route -> {target}")],
        }

    return supervisor


def route_to_specialist(state: AgentState) -> Literal["inventory", "transportation", "supplier"]:
    """Translate the supervisor's chosen route into the next node."""

    route = state["operation"].get("route", SPECIALIST_NAMES[0])
    if route == "transportation":
        return "transportation"
    if route == "supplier":
        return "supplier"
    return "inventory"


def _build_specialist(model: BaseChatModel, name: str) -> Any:
    spec = SPECIALISTS_BY_NAME[name]
    tools = tools_for(name)
    # Binding only this group's tools is the tool-grouping boundary: the model for this
    # node can call its three tools and nothing else.
    bound = model.bind_tools(tools)
    system = SystemMessage(
        content=SPECIALIST_SYSTEM_PROMPT.format(name=spec.name, description=spec.description)
    )

    def specialist(state: AgentState) -> dict[str, Any]:
        query = state["operation"]["query"]
        max_rounds = state["operation"].get("max_tool_rounds", DEFAULT_TOOL_ROUNDS)
        convo: list[BaseMessage] = [system, HumanMessage(content=query)]
        emitted: list[BaseMessage] = []

        for _ in range(max_rounds):
            response = bound.invoke(convo)
            convo.append(response)
            emitted.append(response)
            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
                break
            for call in tool_calls:
                result = run_tool(call["name"], call["args"])
                tool_msg = ToolMessage(content=result, tool_call_id=call["id"])
                convo.append(tool_msg)
                emitted.append(tool_msg)

        final = _text(emitted[-1]) if emitted else ""
        return {
            "operation": {**state["operation"], "specialist": name, "final_answer": final},
            "messages": emitted,
        }

    return specialist


def build_graph(model: BaseChatModel) -> CompiledStateGraph[Any, Any, Any]:
    """Wire supervisor + three specialists with a conditional routing edge."""

    workflow: StateGraph[Any, Any, Any, Any] = StateGraph(AgentState)
    workflow.add_node("supervisor", _build_supervisor(model))
    for name in SPECIALIST_NAMES:
        workflow.add_node(name, _build_specialist(model, name))

    workflow.add_edge(START, "supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        route_to_specialist,
        {name: name for name in SPECIALIST_NAMES},
    )
    for name in SPECIALIST_NAMES:
        workflow.add_edge(name, END)

    return workflow.compile()


def initial_state(query: str, max_tool_rounds: int = DEFAULT_TOOL_ROUNDS) -> AgentState:
    """Seed state: the query plus an empty transcript."""

    return {
        "operation": {"query": query, "max_tool_rounds": max_tool_rounds},
        "messages": [],
    }


def draw_mermaid(model: BaseChatModel | None = None) -> str:
    """Return the graph structure as mermaid text. No API key required."""

    if model is None:
        from .fake import FakeChatModel

        model = FakeChatModel()
    return build_graph(model).get_graph().draw_mermaid()
