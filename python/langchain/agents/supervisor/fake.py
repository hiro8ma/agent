"""A deterministic fake chat model so the supervisor graph runs without an API key.

It plays two roles, told apart by the system prompt:

- Supervisor: keyword-routes the request to one specialist and returns that one word.
- Specialist: on the first turn it emits a tool call for each bound tool whose keywords
  match the request (so the tool-call loop is exercised); on the next turn, once
  ToolMessages are present, it returns a short text answer.

Tools are surfaced via ``bind_tools``; the fake records them so a specialist turn only
ever calls tools from its own group, mirroring how a real model is constrained to the
bound schema.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

# Keyword hints per specialist, used by the supervisor router.
_ROUTE_HINTS: dict[str, tuple[str, ...]] = {
    "inventory": ("stock", "sku", "inventory", "warehouse", "reorder", "replenish", "demand",
                  "forecast", "stockout", "on-hand", "on hand"),
    "transportation": ("ship", "shipment", "transport", "delay", "deliver", "eta", "route",
                        "reroute", "carrier", "freight", "lane", "logistic"),
    "supplier": ("supplier", "vendor", "compliance", "certif", "audit", "sourcing", "score",
                 "scorecard", "alternate", "sanction"),
}

# Keywords that decide which of a specialist's own tools to call.
_TOOL_HINTS: dict[str, tuple[str, ...]] = {
    "check_stock_level": ("stock", "on-hand", "on hand", "stockout", "level", "available"),
    "forecast_demand": ("forecast", "demand", "projection", "trend", "predict"),
    "place_reorder": ("reorder", "replenish", "re-order", "purchase", "po", "order"),
    "track_shipment": ("track", "where", "eta", "delay", "location", "status"),
    "arrange_transport": ("book", "arrange", "transport", "lane", "expedite", "carrier"),
    "reroute_shipment": ("reroute", "route", "around", "disruption", "alternate route"),
    "score_supplier": ("score", "scorecard", "performance", "rating", "on-time", "evaluate"),
    "check_compliance": ("compliance", "compliant", "certif", "audit", "sanction", "iso"),
    "find_alternate_supplier": ("alternate", "backup", "switch", "sourcing", "replace"),
}


class FakeChatModel(BaseChatModel):
    """No-network chat model used for offline supervisor + specialist runs."""

    _bound_tools: list[BaseTool] = []  # noqa: RUF012

    @property
    def _llm_type(self) -> str:
        return "fake-supervisor"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[Any, AIMessage]:
        clone = FakeChatModel()
        clone._bound_tools = [t for t in tools if isinstance(t, BaseTool)]
        return clone

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        system = messages[0].content if messages and isinstance(messages[0].content, str) else ""
        if "supervisor of a supply-chain operations team" in system:
            return self._wrap(self._route(messages))
        return self._specialist_turn(messages)

    @staticmethod
    def _wrap(message: AIMessage) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _route(self, messages: list[BaseMessage]) -> AIMessage:
        text = self._last_human(messages).lower()
        best, best_hits = "inventory", -1
        for name, hints in _ROUTE_HINTS.items():
            hits = sum(1 for h in hints if h in text)
            if hits > best_hits:
                best, best_hits = name, hits
        return AIMessage(content=best)

    def _specialist_turn(self, messages: list[BaseMessage]) -> ChatResult:
        # Once tool results are present, stop calling tools and answer in text.
        if any(isinstance(m, ToolMessage) for m in messages):
            findings = [m.content for m in messages if isinstance(m, ToolMessage)]
            answer = "Findings:\n" + "\n".join(f"- {c}" for c in findings)
            return self._wrap(AIMessage(content=answer))

        text = self._last_human(messages).lower()
        bound = {t.name for t in self._bound_tools}
        calls: list[dict[str, Any]] = []
        for name in bound:
            hints = _TOOL_HINTS.get(name, ())
            if any(h in text for h in hints):
                calls.append({"name": name, "args": _fake_args(name, text), "id": f"call_{name}"})
        if not calls and self._bound_tools:
            first = self._bound_tools[0].name
            calls.append({"name": first, "args": _fake_args(first, text), "id": f"call_{first}"})
        return self._wrap(AIMessage(content="", tool_calls=calls))

    @staticmethod
    def _last_human(messages: list[BaseMessage]) -> str:
        for message in reversed(messages):
            if message.type == "human" and isinstance(message.content, str):
                return message.content
        return ""


def _fake_args(tool_name: str, text: str) -> dict[str, Any]:
    """Build plausible offline args from the request text per tool schema."""

    if tool_name in ("check_stock_level", "forecast_demand", "place_reorder",
                     "find_alternate_supplier"):
        args: dict[str, Any] = {"sku": _grab(text, "sku", "SKU-4471")}
        if tool_name == "place_reorder":
            args["quantity"] = 200
        return args
    if tool_name in ("track_shipment", "reroute_shipment"):
        return {"shipment_id": _grab(text, "shp", "SHP-908")}
    if tool_name == "arrange_transport":
        return {"origin": "WH-NORTH", "destination": "HUB-CENTRAL", "priority": "expedited"}
    if tool_name in ("score_supplier", "check_compliance"):
        return {"supplier_id": _grab(text, "sup", "SUP-12")}
    return {}


def _grab(text: str, prefix: str, default: str) -> str:
    """Pull an id token like 'SKU-4471' from the text, else fall back to a default.

    Requires a digit so a plain domain word (e.g. 'supplier') is not mistaken for an id.
    """

    for token in text.replace(",", " ").split():
        cleaned = token.strip(".").upper()
        if cleaned.lower().startswith(prefix) and any(ch.isdigit() for ch in cleaned):
            return cleaned
    return default
