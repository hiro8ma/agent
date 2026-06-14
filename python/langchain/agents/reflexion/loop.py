"""The Reflexion control loop: act → evaluate → (on failure) reflect → retry.

A plain Python loop, not a LangGraph StateGraph — the cross-attempt episodic memory is
the interesting part here, and a list + buffer expresses it more directly than graph
state would. (The sibling ``agents.reflection`` covers the LangGraph variant.)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from .actor import Actor, FakeActor, LLMActor
from .memory import ReflexionBuffer
from .task import CodingTask, default_task

DEFAULT_MAX_ATTEMPTS = 4


@dataclass
class AttemptRecord:
    """Full trace of one attempt — what was injected, produced, observed, scored."""

    index: int
    injected_reflection: str  # the buffer rendered into the prompt for this attempt
    code: str
    observation: str
    passed: bool
    passed_count: int
    total: int


@dataclass
class ReflexionResult:
    solved: bool
    attempts: list[AttemptRecord] = field(default_factory=list)
    reflections: list[str] = field(default_factory=list)
    used_fake: bool = False


def select_actor(use_fake: bool) -> tuple[Actor, bool]:
    """Return (actor, is_fake).

    Mirrors ``agents.reflection``: fall back to the deterministic offline actor when no
    LLM key is present so the loop runs without credentials.
    """

    if not use_fake and os.environ.get("OPENAI_API_KEY"):
        from core.providers.factory import select_provider

        return LLMActor(select_provider()), False
    return FakeActor(), True


def run_reflexion(
    task: CodingTask | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    use_fake: bool = False,
    memory_file: str | None = None,
) -> ReflexionResult:
    """Run the act→evaluate→reflect loop until success or the attempt cap.

    On each failure a one-line self-reflection is appended to the buffer; the buffer's
    recent slice is injected at the head of the next attempt's prompt. The attempt cap
    is the hard termination guard.
    """

    task = task or default_task()
    actor, is_fake = select_actor(use_fake)
    buffer = ReflexionBuffer(path=memory_file)

    result = ReflexionResult(solved=False, used_fake=is_fake)

    for i in range(1, max_attempts + 1):
        injected = buffer.render()
        code = actor.act(task.prompt, injected)
        evaluation = task.evaluate(code)

        result.attempts.append(
            AttemptRecord(
                index=i,
                injected_reflection=injected,
                code=code,
                observation=evaluation.observation,
                passed=evaluation.passed,
                passed_count=evaluation.passed_count,
                total=evaluation.total,
            )
        )

        if evaluation.passed:
            result.solved = True
            break

        reflection = actor.reflect(task.prompt, code, evaluation.observation)
        buffer.add(reflection)
        result.reflections.append(reflection.strip())

    return result
