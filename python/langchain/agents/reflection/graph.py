"""Plan→Generate→Reflect workflow built on a LangGraph StateGraph.

Flow:

    planner → generator → should_continue ─PASS / cap reached→ END
                              │
                              └─REVISE→ reflector → planner (loop)

The planner decomposes the task into steps, the generator drafts an answer, and the
reflector grades it. A failing grade routes back through the planner with the critique
folded in. An iteration cap guarantees termination regardless of model behaviour.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .prompt import (
    GENERATOR_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    REFLECTOR_SYSTEM_PROMPT,
)

DEFAULT_MAX_ITERATIONS = 3


class ReflectionState(TypedDict):
    """Shared state for the Plan→Generate→Reflect graph.

    `history` uses an `operator.add` reducer so every node appends a trace line and
    the full audit trail accumulates across iterations. The remaining fields use the
    default last-write-wins reducer on purpose: each cycle should overwrite the prior
    plan / draft / critique rather than pile up stale versions.
    """

    task: str
    history: Annotated[list[str], operator.add]
    plan: str
    draft: str
    critique: str
    verdict: str
    iteration: int
    max_iterations: int


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


def _build_planner(model: BaseChatModel) -> Any:
    def planner(state: ReflectionState) -> dict[str, Any]:
        critique = state.get("critique", "")
        if critique:
            user = (
                f"Task:\n{state['task']}\n\n"
                f"Previous plan:\n{state.get('plan', '')}\n\n"
                f"Critique to address:\n{critique}\n\n"
                "Produce a revised plan."
            )
        else:
            user = f"Task:\n{state['task']}\n\nProduce the initial plan."
        plan = _invoke_text(model, PLANNER_SYSTEM_PROMPT, user)
        return {"plan": plan, "history": [f"[plan] {plan}"]}

    return planner


def _build_generator(model: BaseChatModel) -> Any:
    def generator(state: ReflectionState) -> dict[str, Any]:
        user = f"Task:\n{state['task']}\n\nPlan:\n{state['plan']}\n\nWrite the answer."
        draft = _invoke_text(model, GENERATOR_SYSTEM_PROMPT, user)
        # iteration counts completed Generate passes; drives the cap in should_continue.
        return {
            "draft": draft,
            "iteration": state.get("iteration", 0) + 1,
            "history": [f"[draft] {draft}"],
        }

    return generator


def _build_reflector(model: BaseChatModel) -> Any:
    def reflector(state: ReflectionState) -> dict[str, Any]:
        user = f"Task:\n{state['task']}\n\nDraft:\n{state['draft']}\n\nCritique the draft."
        critique = _invoke_text(model, REFLECTOR_SYSTEM_PROMPT, user)
        return {"critique": critique, "history": [f"[reflect] {critique}"]}

    return reflector


def reflector_verdict(critique: str) -> str:
    """Parse the reflector's VERDICT line. Unknown / empty defaults to REVISE."""

    for line in critique.splitlines():
        stripped = line.strip().upper()
        if stripped.startswith("VERDICT:"):
            return "PASS" if "PASS" in stripped else "REVISE"
    return "REVISE"


def should_continue(state: ReflectionState) -> Literal["reflect", "end"]:
    """Decide whether to loop again after a Generate pass.

    Stops when the iteration cap is hit (hard termination guard against a model that
    never returns PASS) or when the latest critique verdict is PASS.
    """

    if state["iteration"] >= state["max_iterations"]:
        return "end"
    if reflector_verdict(state.get("critique", "")) == "PASS":
        return "end"
    return "reflect"


def build_graph(model: BaseChatModel) -> CompiledStateGraph[Any, Any, Any]:
    """Wire planner / generator / reflector into a compiled Runnable."""

    workflow: StateGraph[Any, Any, Any, Any] = StateGraph(ReflectionState)
    workflow.add_node("planner", _build_planner(model))
    workflow.add_node("generator", _build_generator(model))
    workflow.add_node("reflector", _build_reflector(model))

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "generator")
    workflow.add_conditional_edges(
        "generator",
        should_continue,
        {"reflect": "reflector", "end": END},
    )
    workflow.add_edge("reflector", "planner")

    return workflow.compile()


def initial_state(task: str, max_iterations: int = DEFAULT_MAX_ITERATIONS) -> ReflectionState:
    """Seed state for a run. Mutable / accumulating fields start empty."""

    return {
        "task": task,
        "history": [],
        "plan": "",
        "draft": "",
        "critique": "",
        "verdict": "",
        "iteration": 0,
        "max_iterations": max_iterations,
    }


def draw_mermaid(model: BaseChatModel | None = None) -> str:
    """Return the graph structure as mermaid text.

    Works without any API key: a FakeChatModel is used purely to satisfy node
    construction, since the topology is independent of the backend model.
    """

    if model is None:
        from .fake import FakeChatModel

        model = FakeChatModel()
    return build_graph(model).get_graph().draw_mermaid()
