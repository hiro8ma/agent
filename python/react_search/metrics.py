from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from questions import Question
from react import Outcome
from retrieval import normalize


def answer_score(answer: str, question: Question) -> float:
    text = normalize(answer).replace(" ", "")
    hits = sum(1 for alts in question.answers if any(normalize(a) in text for a in alts))
    return hits / len(question.answers)


def evidence_recall(retrieved: set[int], question: Question) -> float:
    hits = sum(1 for alts in question.evidence if retrieved & set(alts))
    return hits / len(question.evidence)


@dataclass(frozen=True)
class Summary:
    n: int
    accuracy: float
    evidence: float
    searches: float
    lookups: float
    calls: float
    tokens: float
    seconds: float
    forced: int
    malformed_questions: int
    malformed_steps: int
    fabricated_questions: int
    fabrications: int
    mismatches: int
    answered_without_search: int


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def summarize(outcomes: Sequence[Outcome]) -> Summary:
    return Summary(
        n=len(outcomes),
        accuracy=mean([answer_score(o.answer, o.question) for o in outcomes]),
        evidence=mean([evidence_recall(o.retrieved, o.question) for o in outcomes]),
        searches=mean([o.searches for o in outcomes]),
        lookups=mean([o.lookups for o in outcomes]),
        calls=mean([o.calls for o in outcomes]),
        tokens=mean([o.tokens for o in outcomes]),
        seconds=mean([o.seconds for o in outcomes]),
        forced=sum(1 for o in outcomes if o.forced),
        malformed_questions=sum(1 for o in outcomes if o.malformed),
        malformed_steps=sum(o.malformed for o in outcomes),
        fabricated_questions=sum(1 for o in outcomes if o.fabrications),
        fabrications=sum(len(o.fabrications) for o in outcomes),
        mismatches=sum(1 for o in outcomes for f in o.fabrications if f.mismatch),
        answered_without_search=sum(
            1 for o in outcomes if o.answer and o.env is not None and o.searches == 0
        ),
    )
