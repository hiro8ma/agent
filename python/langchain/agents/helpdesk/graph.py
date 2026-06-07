"""Plan-and-Execute helpdesk agent built on a LangGraph StateGraph.

Flow:

    inquiry → plan ─→ execute ──→ should_retry ─PASS / cap→ advance ─more→ execute
                                       │                          │
                                       └─RETRY→ execute (loop)    └─done→ synthesize → END

`plan` splits the inquiry into independent subtasks. `execute` handles the current
subtask in three steps (route → retrieve → answer). `should_retry` grades the answer
and either retries the same subtask (bounded by an iteration cap) or advances to the
next one. `synthesize` merges all subtask answers into the final reply.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .prompt import (
    ANSWER_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    REFLECT_SYSTEM_PROMPT,
    ROUTER_SYSTEM_PROMPT,
    SYNTHESIZE_SYSTEM_PROMPT,
)
from .tools import search_manual, search_qa

DEFAULT_MAX_RETRIES = 2


class HelpdeskState(TypedDict):
    """Shared state for the Plan-and-Execute graph.

    Reducers are chosen per field on purpose:
    - `sub_answers` and `history` use `operator.add` so each completed subtask appends
      its result and the audit trail accumulates across the whole run.
    - `current_subtask_index` and `iteration` use last-write-wins: they are cursors,
      not logs, so a node overwrites them rather than accumulating stale values.
    - `final_answer` is written once by `synthesize`.
    """

    query: str
    subtasks: list[str]
    sub_answers: Annotated[list[str], operator.add]
    current_subtask_index: int
    iteration: int
    max_retries: int
    final_answer: str
    history: Annotated[list[str], operator.add]
    # Per-subtask scratch (last-write-wins): the in-flight draft answer and its critique.
    draft: str
    critique: str


def _invoke_text(model: BaseChatModel, system: str, user: str) -> str:
    response = model.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    content = response.content
    if isinstance(content, str):
        return content.strip()
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and isinstance(part.get("text"), str):
            parts.append(part["text"])
    return "\n".join(parts).strip()


def _build_plan(model: BaseChatModel) -> Any:
    def plan(state: HelpdeskState) -> dict[str, Any]:
        text = _invoke_text(model, PLANNER_SYSTEM_PROMPT, f"Inquiry:\n{state['query']}")
        subtasks = [line.strip("-• \t") for line in text.splitlines() if line.strip()]
        if not subtasks:
            subtasks = [state["query"]]
        return {
            "subtasks": subtasks,
            "current_subtask_index": 0,
            "iteration": 0,
            "history": [f"[plan] {len(subtasks)} subtask(s): {subtasks}"],
        }

    return plan


def _route(model: BaseChatModel, subtask: str) -> str:
    choice = _invoke_text(model, ROUTER_SYSTEM_PROMPT, f"Subtask:\n{subtask}").lower()
    # Default to keyword search when the model returns anything unexpected.
    return "search_qa" if "search_qa" in choice else "search_manual"


def _build_execute(model: BaseChatModel) -> Any:
    def execute(state: HelpdeskState) -> dict[str, Any]:
        index = state["current_subtask_index"]
        subtask = state["subtasks"][index]

        # 2.1 tool selection — delegated to the LLM via tool descriptions.
        tool_name = _route(model, subtask)
        # 2.2 tool execution.
        context = search_qa(subtask) if tool_name == "search_qa" else search_manual(subtask)
        # 2.3 answer generation. The user message carries only this subtask's context,
        # so subtasks stay isolated and never leak each other's retrieval results.
        answer = _invoke_text(
            model,
            ANSWER_SYSTEM_PROMPT,
            f"Subtask:\n{subtask}\n\nRetrieved context:\n{context}",
        )
        return {
            "iteration": state["iteration"] + 1,
            "history": [
                f"[execute#{index}] tool={tool_name} try={state['iteration'] + 1}",
                f"[answer#{index}] {answer}",
            ],
            # Stash the draft for the reflector; committed to sub_answers only on advance.
            "draft": answer,
        }

    return execute


def _build_reflect(model: BaseChatModel) -> Any:
    def reflect(state: HelpdeskState) -> dict[str, Any]:
        index = state["current_subtask_index"]
        subtask = state["subtasks"][index]
        draft = state.get("draft", "")
        critique = _invoke_text(
            model,
            REFLECT_SYSTEM_PROMPT,
            f"Subtask:\n{subtask}\n\nAnswer:\n{draft}",
        )
        return {"critique": critique, "history": [f"[reflect#{index}] {critique}"]}

    return reflect


def reflect_verdict(critique: str) -> str:
    """Parse the reflector's VERDICT line. Unknown / empty defaults to RETRY."""

    for line in critique.splitlines():
        stripped = line.strip().upper()
        if stripped.startswith("VERDICT:"):
            return "PASS" if "PASS" in stripped else "RETRY"
    return "RETRY"


def should_retry(state: HelpdeskState) -> Literal["retry", "advance"]:
    """After reflect: retry the same subtask or advance to the next one.

    The iteration cap is the hard termination guard — it stops a subtask from looping
    forever when the model never returns PASS.
    """

    if state["iteration"] >= state["max_retries"] + 1:
        return "advance"
    if reflect_verdict(state.get("critique", "")) == "PASS":
        return "advance"
    return "retry"


def _advance(state: HelpdeskState) -> dict[str, Any]:
    """Commit the accepted draft and move the cursor; reset the per-subtask iteration."""

    index = state["current_subtask_index"]
    draft = state.get("draft", "")
    return {
        "sub_answers": [f"[{state['subtasks'][index]}]\n{draft}"],
        "current_subtask_index": index + 1,
        "iteration": 0,
        "history": [f"[advance] subtask {index} committed"],
    }


def has_more(state: HelpdeskState) -> Literal["execute", "synthesize"]:
    """Loop back to execute the next subtask, or synthesize when all are done."""

    if state["current_subtask_index"] < len(state["subtasks"]):
        return "execute"
    return "synthesize"


def _build_synthesize(model: BaseChatModel) -> Any:
    def synthesize(state: HelpdeskState) -> dict[str, Any]:
        joined = "\n\n".join(state["sub_answers"])
        final = _invoke_text(
            model,
            SYNTHESIZE_SYSTEM_PROMPT,
            f"Original inquiry:\n{state['query']}\n\nSubtask answers:\n{joined}",
        )
        return {"final_answer": final, "history": ["[synthesize] final answer produced"]}

    return synthesize


def build_graph(model: BaseChatModel) -> CompiledStateGraph[Any, Any, Any]:
    """Wire plan / execute / reflect / advance / synthesize into a compiled Runnable."""

    workflow: StateGraph[Any, Any, Any, Any] = StateGraph(HelpdeskState)
    workflow.add_node("plan", _build_plan(model))
    workflow.add_node("execute", _build_execute(model))
    workflow.add_node("reflect", _build_reflect(model))
    workflow.add_node("advance", _advance)
    workflow.add_node("synthesize", _build_synthesize(model))

    workflow.add_edge(START, "plan")
    workflow.add_edge("plan", "execute")
    workflow.add_edge("execute", "reflect")
    workflow.add_conditional_edges(
        "reflect",
        should_retry,
        {"retry": "execute", "advance": "advance"},
    )
    workflow.add_conditional_edges(
        "advance",
        has_more,
        {"execute": "execute", "synthesize": "synthesize"},
    )
    workflow.add_edge("synthesize", END)

    return workflow.compile()


def initial_state(query: str, max_retries: int = DEFAULT_MAX_RETRIES) -> HelpdeskState:
    """Seed state for a run. Accumulating fields start empty; cursors start at 0."""

    return {
        "query": query,
        "subtasks": [],
        "sub_answers": [],
        "current_subtask_index": 0,
        "iteration": 0,
        "max_retries": max_retries,
        "final_answer": "",
        "history": [],
        "draft": "",
        "critique": "",
    }


def draw_mermaid(model: BaseChatModel | None = None) -> str:
    """Return the graph structure as mermaid text. No API key required."""

    if model is None:
        from .fake import FakeChatModel

        model = FakeChatModel()
    return build_graph(model).get_graph().draw_mermaid()
