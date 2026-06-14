"""SummarizerAgent — a sample A2A peer advertising a ``summarizeText`` capability.

The summary is a deterministic, offline fake (first sentence + word-count
compression), mirroring the repo's ``fake.py`` convention so the demo runs with no
API key. Swapping ``_fake_summarize`` for a real LLM call leaves the card, server,
and JSON-RPC contract untouched — A2A peers are judged by their card, not their
implementation.
"""

from __future__ import annotations

import re
from typing import Any

from .card import AgentCard, Schema
from .errors import INVALID_PARAMS, A2AError
from .server import A2AServer

CAPABILITY = "summarizeText"

_CARD = AgentCard(
    identity="summarizer",
    version="1.2.0",
    capabilities=[CAPABILITY],
    schemas={
        CAPABILITY: Schema(
            input={"text": "string", "maxWords": "integer?"},
            output={"summary": "string", "wordCount": "integer"},
        )
    },
    endpoint="http://localhost/agents/summarizer",  # dummy: in-process in this demo
    auth_methods=["none"],
)


def _first_sentence(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return parts[0] if parts and parts[0] else text.strip()


def _fake_summarize(text: str, max_words: int) -> str:
    """Lead sentence, then truncate to ``max_words`` — deterministic, no network."""
    lead = _first_sentence(text)
    words = lead.split()
    if len(words) <= max_words:
        return lead
    return " ".join(words[:max_words]).rstrip(".,;:") + " …"


def _summarize_text(params: dict[str, Any]) -> dict[str, Any]:
    text = params.get("text")
    if not isinstance(text, str) or not text.strip():
        raise A2AError("params.text must be a non-empty string", code=INVALID_PARAMS)
    max_words = params.get("maxWords", 12)
    if not isinstance(max_words, int) or max_words <= 0:
        raise A2AError("params.maxWords must be a positive integer", code=INVALID_PARAMS)

    summary = _fake_summarize(text, max_words)
    return {"summary": summary, "wordCount": len(summary.split())}


def build_server() -> A2AServer:
    """Construct the hosted summarizer (card + capability handler)."""
    return A2AServer(card=_CARD, handlers={CAPABILITY: _summarize_text})
