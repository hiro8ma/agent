"""Experience-learning loop: extract insights, distil them into a small rule list,
promote/demote on KPI outcomes, and reflect on the survivors.

The cycle for one observation:

    report + KPI flag
        → extract_insight        (LLM: report -> one-line lesson)
        → distill_insight        (LLM: AGREE / REMOVE / EDIT / ADD vs the list)
        → promote_or_demote      (rule based: success promotes, failure demotes)

Run it over a stream of observations and the rule book converges to a few
high-confidence (promoted) rules while contradicted ones are removed and
underperformers decay (demoted). ``reflect`` then turns the promoted rules into
the next experiment to try.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from .prompt import (
    DISTILL_SYSTEM_PROMPT,
    INSIGHT_SYSTEM_PROMPT,
    REFLECT_SYSTEM_PROMPT,
)
from .rule import Rule, RuleBook

_OP_RE = re.compile(
    r"^\s*(AGREE|REMOVE|EDIT|ADD)\s*(\d+)?\s*(?:=\s*(.*))?\s*$",
    re.IGNORECASE,
)


@dataclass
class Observation:
    """One past run: what was done and whether it hit its KPI target."""

    report: str
    success: bool

    def as_prompt(self) -> str:
        flag = "SUCCESS" if self.success else "FAILURE"
        return f"Report ({flag}): {self.report}"


@dataclass
class DistillResult:
    """Outcome of distilling one insight against the current rule book."""

    insight: str
    op: str
    rule: Rule | None
    note: str


def select_model(use_fake: bool) -> tuple[BaseChatModel, bool]:
    """Return (model, is_fake). Falls back to the offline fake when no key is set."""

    if not use_fake and os.environ.get("OPENAI_API_KEY"):
        from core.providers.factory import select_provider

        return select_provider(), False

    from .fake import FakeChatModel

    return FakeChatModel(), True


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


def extract_insight(model: BaseChatModel, obs: Observation) -> str:
    """Turn one observation into a single reusable lesson."""

    return _invoke_text(model, INSIGHT_SYSTEM_PROMPT, obs.as_prompt())


def _parse_op(raw: str) -> tuple[str, int | None, str]:
    """Parse the distiller's single-line decision into (op, rule_id, text).

    Unknown / unparseable output defaults to ADD so a new insight is never lost.
    """

    first_line = raw.strip().splitlines()[0] if raw.strip() else ""
    m = _OP_RE.match(first_line)
    if not m:
        return "ADD", None, ""
    op = m.group(1).upper()
    rule_id = int(m.group(2)) if m.group(2) else None
    text = (m.group(3) or "").strip()
    return op, rule_id, text


def distill_insight(model: BaseChatModel, book: RuleBook, insight: str) -> DistillResult:
    """Compare one insight to the rule book and apply one of the four ops.

    AGREE keeps a matching rule, REMOVE deletes a contradicted/redundant one,
    EDIT rewords a near-match, ADD appends a genuinely new rule. Biasing toward
    AGREE/EDIT/REMOVE is what keeps the list small under a drifting world.
    """

    user = f"Existing rules:\n{book.render()}\n\nCandidate: {insight}"
    raw = _invoke_text(model, DISTILL_SYSTEM_PROMPT, user)
    op, rule_id, text = _parse_op(raw)

    if op == "AGREE" and rule_id is not None:
        rule = book.agree(rule_id)
        return DistillResult(insight, "AGREE", rule, f"kept rule {rule_id}")
    if op == "REMOVE" and rule_id is not None:
        removed = book.remove(rule_id)
        return DistillResult(insight, "REMOVE", None, f"removed rule {rule_id}={removed}")
    if op == "EDIT" and rule_id is not None and text:
        rule = book.edit(rule_id, text)
        return DistillResult(insight, "EDIT", rule, f"reworded rule {rule_id}")

    new_text = text or insight
    rule = book.add(new_text)
    return DistillResult(insight, "ADD", rule, f"added rule {rule.id}")


def promote_or_demote(book: RuleBook, rule: Rule | None, success: bool) -> Rule | None:
    """Apply the KPI outcome to the rule that was just touched.

    Success promotes (confidence up), failure demotes (decay, reversible). Rules
    removed by distillation are ``None`` and skipped.
    """

    if rule is None:
        return None
    return book.promote(rule.id) if success else book.demote(rule.id)


def learn(
    model: BaseChatModel,
    book: RuleBook,
    observations: list[Observation],
) -> list[DistillResult]:
    """Run the full extract → distil → promote/demote cycle over observations."""

    results: list[DistillResult] = []
    for obs in observations:
        insight = extract_insight(model, obs)
        result = distill_insight(model, book, insight)
        promote_or_demote(book, result.rule, obs.success)
        results.append(result)
    return results


def reflect(model: BaseChatModel, book: RuleBook) -> str:
    """Propose the next high-impact experiment from the promoted rules."""

    promoted = book.promoted()
    if not promoted:
        body = "(no promoted rules yet)"
    else:
        body = "\n".join(f"{r.id}. {r.text}" for r in promoted)
    user = f"Highest-confidence rules:\n{body}"
    return _invoke_text(model, REFLECT_SYSTEM_PROMPT, user)
