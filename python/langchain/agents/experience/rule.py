"""Rule list with a three-tier lifecycle (active / promoted / demoted).

A *rule* is a distilled insight — a short, reusable lesson extracted from past
runs. The point of keeping rules instead of raw reports is non-stationarity:
the world drifts, so the agent must keep a *small* set of currently-true rules
and let stale ones decay rather than accumulate forever.

Three tiers encode confidence over time:

- ``active``   — newly added or edited; not yet proven across runs.
- ``promoted`` — a rule that held up when its KPI target was met (success).
                 These are the high-confidence rules the agent reflects on.
- ``demoted``  — a rule that was in force when the KPI target was missed
                 (failure). Kept, not deleted, because a drifting world may
                 make it relevant again — demotion is reversible decay, REMOVE
                 is permanent deletion for contradictions / duplicates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Tier(StrEnum):
    ACTIVE = "active"
    PROMOTED = "promoted"
    DEMOTED = "demoted"


@dataclass
class Rule:
    """A single distilled insight with a confidence score and tier.

    ``score`` is a running signed counter: +1 each time the rule is promoted on
    a success, -1 each time it is demoted on a failure. It drives the tier so a
    rule that keeps failing sinks even if it was once promoted.
    """

    id: int
    text: str
    tier: Tier = Tier.ACTIVE
    score: int = 0
    history: list[str] = field(default_factory=list)

    def log(self, entry: str) -> None:
        self.history.append(entry)


class RuleBook:
    """Mutable, id-stable collection of rules.

    Ids never get reused: EDIT keeps an id, REMOVE retires it, ADD always takes
    the next fresh id. That stability lets the four ops (AGREE / REMOVE / EDIT /
    ADD) reference rules unambiguously across distillation rounds.
    """

    def __init__(self) -> None:
        self._rules: dict[int, Rule] = {}
        self._next_id: int = 1

    def add(self, text: str) -> Rule:
        rule = Rule(id=self._next_id, text=text)
        rule.log("ADD")
        self._rules[rule.id] = rule
        self._next_id += 1
        return rule

    def get(self, rule_id: int) -> Rule | None:
        return self._rules.get(rule_id)

    def edit(self, rule_id: int, text: str) -> Rule | None:
        rule = self._rules.get(rule_id)
        if rule is None:
            return None
        rule.text = text
        # EDIT resets confidence: the claim changed, so prior promotions no
        # longer apply to the new wording.
        rule.tier = Tier.ACTIVE
        rule.score = 0
        rule.log(f"EDIT -> {text}")
        return rule

    def remove(self, rule_id: int) -> bool:
        return self._rules.pop(rule_id, None) is not None

    def agree(self, rule_id: int) -> Rule | None:
        rule = self._rules.get(rule_id)
        if rule is not None:
            rule.log("AGREE")
        return rule

    def promote(self, rule_id: int) -> Rule | None:
        rule = self._rules.get(rule_id)
        if rule is None:
            return None
        rule.score += 1
        rule.tier = Tier.PROMOTED
        rule.log(f"PROMOTE (score={rule.score})")
        return rule

    def demote(self, rule_id: int) -> Rule | None:
        rule = self._rules.get(rule_id)
        if rule is None:
            return None
        rule.score -= 1
        rule.tier = Tier.DEMOTED
        rule.log(f"DEMOTE (score={rule.score})")
        return rule

    def all(self) -> list[Rule]:
        return sorted(self._rules.values(), key=lambda r: r.id)

    def by_tier(self, tier: Tier) -> list[Rule]:
        return [r for r in self.all() if r.tier == tier]

    def promoted(self) -> list[Rule]:
        return self.by_tier(Tier.PROMOTED)

    def render(self) -> str:
        """One numbered line per rule, used both for display and as LLM context."""

        if not self._rules:
            return "(empty rule book)"
        lines = []
        for rule in self.all():
            lines.append(f"{rule.id}. [{rule.tier.value}|score={rule.score}] {rule.text}")
        return "\n".join(lines)
