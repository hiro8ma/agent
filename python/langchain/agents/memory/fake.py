"""A deterministic fake chat model for the memory demo.

It only needs to prove one thing: whether the model can answer "what's my name?".
It scans the messages it is handed for an "I'm <name>" introduction and echoes the
name back. With a checkpointer the prior turn is replayed into `messages`, so the name
is present and the model answers it. Without a checkpointer the model is handed only the
current turn, the introduction is gone, and the same model cannot answer — which is the
whole point of the demo.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

_INTRO = re.compile(r"\bi'?m\s+([A-Za-z][A-Za-z'-]*)", re.IGNORECASE)


def _find_name(messages: list[BaseMessage]) -> str | None:
    for msg in messages:
        if isinstance(msg.content, str):
            match = _INTRO.search(msg.content)
            if match:
                return match.group(1)
    return None


class FakeChatModel(BaseChatModel):
    """No-network chat model whose only skill is recalling an introduced name."""

    @property
    def _llm_type(self) -> str:
        return "fake-memory"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        last = messages[-1].content if messages else ""
        text = last if isinstance(last, str) else ""

        if "name" in text.lower():
            name = _find_name(messages)
            reply = f"Your name is {name}." if name else "I don't know your name yet."
        else:
            name = _find_name(messages)
            reply = f"Nice to meet you, {name}!" if name else "Hello."

        message = AIMessage(content=reply)
        return ChatResult(generations=[ChatGeneration(message=message)])
