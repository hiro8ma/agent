"""Shared selection result type and the offline fake selector.

The fake selector is keyword-based and deterministic so all three strategies reach the
same tool for the same query without any LLM key. Real LLM tool-calling is wired in the
native strategy via bind_tools; semantic and hierarchical reuse this keyword fallback
for parameter/group decisions when no key is present.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Selection:
    """Outcome of one strategy run: which tool, with what args, at what LLM cost."""

    strategy: str
    tool_name: str
    args: dict[str, object]
    llm_calls: int
    considered: list[str] = field(default_factory=list)
    result: str = ""
    is_fake: bool = True


# Keyword hints map a query to a tool. Ordered: first match wins.
_HINTS: list[tuple[tuple[str, ...], str]] = [
    (("solve", "equation", "=", "x+", "calculate", "evaluate"), "solve_equation"),
    (("convert", "km", "miles", "celsius", "fahrenheit", "kg"), "convert_units"),
    (("webhook", "workflow", "trigger", "automation", "zapier"), "run_webhook"),
    (("schedule", "cron", "recurring", "every day", "nightly"), "schedule_job"),
    (("slack", "channel", "#", "notify", "message", "post"), "send_chat_message"),
    (("email", "mail", "@", "subject"), "send_email"),
]


def fake_pick_tool(query: str, candidates: list[str]) -> str:
    """Pick a tool name from candidates by keyword hints (deterministic)."""

    lowered = query.lower()
    for terms, name in _HINTS:
        if name in candidates and any(t in lowered for t in terms):
            return name
    return candidates[0]


def fake_args(query: str, tool_name: str) -> dict[str, object]:
    """Produce plausible args for a tool from the raw query (deterministic)."""

    if tool_name == "solve_equation":
        return {"expression": query}
    if tool_name == "convert_units":
        return {"quantity": query}
    if tool_name == "run_webhook":
        return {"workflow": query, "payload": ""}
    if tool_name == "schedule_job":
        return {"job": query, "cron": "0 9 * * *"}
    if tool_name == "send_chat_message":
        channel = next((w for w in query.split() if w.startswith("#")), "#general")
        return {"channel": channel, "text": query}
    if tool_name == "send_email":
        return {"to": "team@example.com", "subject": query[:40], "body": query}
    return {}
