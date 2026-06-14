"""Actors and the self-reflection step.

Two backends share one interface:

* ``LLMActor`` — a real ChatModel writes code and verbalises reflections.
* ``FakeActor`` — deterministic, offline. Crucially it emits a *buggy* solution while
  the injected reflection block is empty and the *correct* one once a reflection is
  present. That makes the effect of reflection-injection visible without any network:
  attempt 1 (no memory) fails, attempt 2 (memory injected) passes.
"""

from __future__ import annotations

from typing import Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from .prompt import ACTOR_SYSTEM_PROMPT, REFLECTOR_SYSTEM_PROMPT


class Actor(Protocol):
    def act(self, task_prompt: str, reflections: str) -> str:
        """Return code for the task. `reflections` is the rendered buffer (may be '')."""

    def reflect(self, task_prompt: str, code: str, observation: str) -> str:
        """Return one line of self-critique after a failed attempt."""


def _strip_fences(text: str) -> str:
    """Remove ```python fences a model may add despite instructions."""

    lines = [ln for ln in text.splitlines() if not ln.strip().startswith("```")]
    return "\n".join(lines).strip()


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


class LLMActor:
    """Actor backed by a real chat model."""

    def __init__(self, model: BaseChatModel) -> None:
        self._model = model

    def act(self, task_prompt: str, reflections: str) -> str:
        user = (f"{reflections}\n\n" if reflections else "") + task_prompt
        return _strip_fences(_invoke_text(self._model, ACTOR_SYSTEM_PROMPT, user))

    def reflect(self, task_prompt: str, code: str, observation: str) -> str:
        user = (
            f"Task:\n{task_prompt}\n\n"
            f"Your code:\n{code}\n\n"
            f"Evaluator observation:\n{observation}\n\n"
            "Write one line of self-critique for the next attempt."
        )
        return _invoke_text(self._model, REFLECTOR_SYSTEM_PROMPT, user)


# Deterministic outputs for the offline fake. The buggy version sums ALL numbers
# (ignores the even-only requirement); the fixed version filters on `n % 2 == 0`.
_BUGGY_CODE = "def solve(nums: list[int]) -> int:\n    return sum(nums)"
_FIXED_CODE = "def solve(nums: list[int]) -> int:\n    return sum(n for n in nums if n % 2 == 0)"
_TEMPLATE_REFLECTION = (
    "I summed every element; the task wants only even numbers, "
    "so filter with `n % 2 == 0` before summing."
)


class FakeActor:
    """Offline actor that only succeeds once a reflection has been injected.

    This is the key to the demo: behaviour is a pure function of whether the
    reflection buffer is non-empty, so the log shows learning, not randomness.
    """

    def act(self, task_prompt: str, reflections: str) -> str:
        return _FIXED_CODE if reflections.strip() else _BUGGY_CODE

    def reflect(self, task_prompt: str, code: str, observation: str) -> str:
        return _TEMPLATE_REFLECTION
