"""experience — learn from past runs by distilling reports into a small rule list.

Where ``reflection`` grades a single answer in-loop and ``memory`` stores raw
events, this package closes the longer feedback loop: it extracts a one-line
insight from each finished run, distils it into a *small* book of rules via four
ops (AGREE / REMOVE / EDIT / ADD), and tracks confidence with a three-tier
lifecycle (active / promoted / demoted) driven by whether each run hit its KPI.

The result is a handful of high-confidence rules an agent can reflect on to pick
its next experiment — and that stay current as the world drifts, because losing
rules decay (demote) and contradicted ones are deleted (REMOVE).

``rule.py``    — Rule / RuleBook with the three-tier lifecycle and four ops.
``distill.py`` — extract_insight → distill_insight → promote/demote → reflect.
``prompt.py``  — system prompts for the three LLM roles.
``fake.py``    — deterministic offline model so the loop runs without a key.
"""

from .distill import (
    DistillResult,
    Observation,
    distill_insight,
    extract_insight,
    learn,
    promote_or_demote,
    reflect,
    select_model,
)
from .rule import Rule, RuleBook, Tier

__all__ = [
    "DistillResult",
    "Observation",
    "Rule",
    "RuleBook",
    "Tier",
    "distill_insight",
    "extract_insight",
    "learn",
    "promote_or_demote",
    "reflect",
    "select_model",
]
