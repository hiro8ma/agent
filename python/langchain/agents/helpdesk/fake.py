"""A deterministic fake chat model so the helpdesk graph runs without an API key.

It distinguishes the five roles (plan / route / answer / reflect / synthesize) by the
system prompt. The reflector returns RETRY once then PASS, so the iteration cap and the
PASS short-circuit are both exercised. Routing is keyword-based and deterministic.
"""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

# Terms that suggest an exact-match (keyword) lookup over semantic similarity.
_KEYWORD_HINTS = ("error", "code", "e-", "version", "v3", "scope", "api", "key", "xyz")


class FakeChatModel(BaseChatModel):
    """No-network chat model used for offline graph runs and mermaid rendering."""

    retries_before_pass: int = 1

    _retry_counts: dict[str, int] = PrivateAttr(default_factory=dict)

    @property
    def _llm_type(self) -> str:
        return "fake-helpdesk"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        system = messages[0].content if messages and isinstance(messages[0].content, str) else ""
        user = messages[-1].content if messages and isinstance(messages[-1].content, str) else ""

        if "You are the planner" in system:
            text = self._plan(user)
        elif "route a single helpdesk subtask" in system:
            text = self._route(user)
        elif "answer a single helpdesk subtask" in system:
            text = "Based on the retrieved context, here is the resolution for the subtask."
        elif "verify whether an answer" in system:
            text = self._reflect(user)
        elif "You are the synthesizer" in system:
            text = "Combined final answer covering every subtask of the inquiry."
        else:
            text = "ok"

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    @staticmethod
    def _plan(user: str) -> str:
        """Split the inquiry into subtasks on conjunctions.

        Handles Japanese as well as English. Without the Japanese separators a
        multi-part Japanese inquiry collapses into one subtask, which hides the
        parallel fan-out entirely — the graph looks correct but exercises nothing.
        """

        body = user.split("Inquiry:", 1)[-1].strip()
        for sep in (" and ", "、と", "と、", "および", "ならびに", "。"):
            body = body.replace(sep, "\n")
        parts = [p.strip("と 　") for p in body.split("\n") if p.strip("と 　")]
        return "\n".join(parts) if parts else body

    @staticmethod
    def _route(user: str) -> str:
        lowered = user.lower()
        if any(hint in lowered for hint in _KEYWORD_HINTS):
            return "search_manual"
        return "search_qa"

    def _reflect(self, user: str) -> str:
        """Judge the draft the way the reflect prompt asks.

        The prompt says an answer that could not find the information must be
        marked RETRY. A counter-only reflector passes after N tries regardless of
        what the tools returned, so an empty retrieval still ends as "completed" —
        the graph looks healthy while every subtask answered from nothing.

        Checking the retrieved context here is what makes the not-found path
        reachable at all.
        """

        if "(no matching documents)" in user:
            return "VERDICT: RETRY\nNothing was retrieved; try the other tool or different terms."

        # The caller may label the draft "Answer:" or "Draft answer:".
        # Splitting on the wrong label leaves the whole prompt as the key, so the
        # retry counter never matches and every attempt returns RETRY.
        body = user.split("Subtask:", 1)[-1]
        for label in ("Retrieved context:", "Draft answer:", "Answer:"):
            body = body.split(label, 1)[0]
        subtask = body.strip()
        count = self._retry_counts.get(subtask, 0)
        if count < self.retries_before_pass:
            self._retry_counts[subtask] = count + 1
            return "VERDICT: RETRY\nTry the other retrieval tool for a closer match."
        return "VERDICT: PASS\n"
