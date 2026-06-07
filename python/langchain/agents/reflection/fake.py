"""A deterministic fake chat model so the graph runs without an API key.

It mimics the three roles by inspecting the system prompt and returns a REVISE
verdict for the first two reflections, then PASS — exercising the loop and proving
the iteration cap and the PASS short-circuit both terminate the graph.
"""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class FakeChatModel(BaseChatModel):
    """No-network chat model used for offline graph runs and mermaid rendering."""

    reflections_before_pass: int = 2

    _reflection_count: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-reflection"

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

        if "You are the planner" in system:
            text = "1. Outline the answer\n2. Fill in details\n3. Review"
        elif "You are the generator" in system:
            text = "Draft answer following the plan."
        elif "You are the reflector" in system:
            if self._reflection_count < self.reflections_before_pass:
                self._reflection_count += 1
                text = "VERDICT: REVISE\n1. Add a concrete example."
            else:
                text = "VERDICT: PASS"
        else:
            text = "ok"

        message = AIMessage(content=text)
        return ChatResult(generations=[ChatGeneration(message=message)])
