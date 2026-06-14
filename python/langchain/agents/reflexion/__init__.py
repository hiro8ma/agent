"""reflexion — verbal self-reflection loop (Reflexion-style episodic memory).

Unlike ``agents.reflection`` (an intra-attempt Plan→Generate→Reflect graph), this
module runs *cross-attempt* learning: an objective evaluator grades each attempt, a
failure is verbalised into a short reflection, the reflection is stored in an
episodic buffer, and the next attempt is re-run with the buffer injected at the head
of the prompt. No weights change — the agent learns purely in-context (nonparametric).
"""

from .loop import (
    DEFAULT_MAX_ATTEMPTS,
    AttemptRecord,
    ReflexionResult,
    run_reflexion,
)
from .memory import ReflexionBuffer
from .task import CodingTask, default_task

__all__ = [
    "DEFAULT_MAX_ATTEMPTS",
    "AttemptRecord",
    "CodingTask",
    "ReflexionBuffer",
    "ReflexionResult",
    "default_task",
    "run_reflexion",
]
