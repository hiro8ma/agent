"""A deterministic fake chat model so the experience loop runs without an API key.

It mimics three roles by inspecting the system prompt:

- insight extraction : turn a SUCCESS/FAILURE report into a one-line lesson.
- distillation       : pick one of AGREE / REMOVE / EDIT / ADD against the list.
- reflection         : name the next experiment from the promoted rules.

The distillation logic is intentionally simple but exercises every op: it keys
off the lever word shared between the candidate and an existing rule so the demo
shows AGREE (same lever, same sign), EDIT (same lever, refined), REMOVE (same
lever, opposite sign = contradiction) and ADD (unseen lever).
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

# Levers the demo reports talk about; the first match becomes the insight's topic.
_LEVERS = (
    "traffic",
    "email",
    "cart",
    "order value",
    "sign-up",
)


def _topic(text: str) -> str:
    low = text.lower()
    for lever in _LEVERS:
        if lever in low:
            return lever
    return "kpi"


def _is_success(text: str) -> bool:
    return "success" in text.lower()


def _insight(report: str) -> str:
    topic = _topic(report)
    if _is_success(report):
        return f"Invest more in {topic}; it reliably moves the KPI up."
    return f"Avoid the current {topic} approach; it failed to move the KPI."


def _existing_rules(user: str) -> list[tuple[int, str]]:
    rules: list[tuple[int, str]] = []
    for line in user.splitlines():
        m = re.match(r"\s*(\d+)\.\s+(.*)", line)
        if m:
            rules.append((int(m.group(1)), m.group(2)))
    return rules


def _candidate(user: str) -> str:
    for line in user.splitlines():
        if line.lower().startswith("candidate:"):
            return line.split(":", 1)[1].strip()
    return ""


def _distill_decision(user: str) -> str:
    candidate = _candidate(user)
    cand_topic = _topic(candidate)
    cand_pos = "invest" in candidate.lower()

    for rid, text in _existing_rules(user):
        # strip the leading "[tier|score=n] " bookkeeping the rule book renders.
        body = re.sub(r"^\[[^\]]*\]\s*", "", text)
        if _topic(body) != cand_topic:
            continue
        rule_pos = "invest" in body.lower()
        if rule_pos != cand_pos:
            # same lever, opposite direction -> the old rule is contradicted.
            return f"REMOVE{rid}"
        if body.strip() == candidate.strip():
            return f"AGREE{rid}"
        # same lever, same direction, different wording -> refine in place.
        return f"EDIT{rid}={candidate}"
    return f"ADD={candidate}"


def _reflect(user: str) -> str:
    rules = _existing_rules(user)
    if rules:
        body = re.sub(r"^\[[^\]]*\]\s*", "", rules[0][1])
        return f"Double down on the proven lever: {body}"
    return "Run a broad probe across all levers to find one that moves the KPI."


class FakeChatModel(BaseChatModel):
    """No-network chat model used for offline experience-distillation runs."""

    @property
    def _llm_type(self) -> str:
        return "fake-experience"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        system = ""
        if messages and isinstance(messages[0].content, str):
            system = messages[0].content
        user = ""
        if len(messages) > 1 and isinstance(messages[-1].content, str):
            user = messages[-1].content

        if "distill experience into reusable insights" in system:
            text = _insight(user)
        elif "maintain a small, high-quality list" in system:
            text = _distill_decision(user)
        elif "growth strategist" in system:
            text = _reflect(user)
        else:
            text = "ok"

        message = AIMessage(content=text)
        return ChatResult(generations=[ChatGeneration(message=message)])
