"""Lightweight feedback sink for the web-research agent (Good/Bad ratings).

Maps two of the four agent UX measurement axes onto a durable signal:
- "正確さ&品質" (accuracy / quality): a Bad rating flags an answer the operator
  judged wrong or low quality, so retrieval and synthesis can be tuned.
- "満足度" (satisfaction): the good/bad split plus the optional comment is the
  raw thumbs-up rate the agent is ultimately judged on.

The other two axes (タスク完了率 / 介入度) are observable from the run itself
(did write_file succeed, how many APPROVE/DENY interrupts fired) and are out of
scope here.

Records are appended as one JSON object per line to
``.feedback/feedback.jsonl`` under the langchain package root. Append-only JSONL
keeps writes atomic-enough for a single-operator GUI and trivially greppable.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Literal

Rating = Literal["good", "bad"]

# .../langchain/agents/web_researcher/feedback.py -> .../langchain
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent
FEEDBACK_DIR = _PACKAGE_ROOT / ".feedback"
FEEDBACK_FILE = FEEDBACK_DIR / "feedback.jsonl"

_SUMMARY_LEN = 280


def record_feedback(
    query: str,
    answer: str,
    rating: Rating,
    comment: str = "",
    *,
    timestamp: float | None = None,
) -> dict[str, object]:
    """Append one feedback record and return the row that was written.

    Args:
        query: The research topic the operator asked for.
        answer: The agent's final answer; stored as a truncated summary plus a
            stable sha256 so duplicate answers can be grouped without keeping the
            full text.
        rating: "good" or "bad" thumbs verdict.
        comment: Optional free-text note from the operator.
        timestamp: Unix seconds; defaults to time.time() at call time.
    """

    if rating not in ("good", "bad"):
        raise ValueError(f"rating must be 'good' or 'bad', got {rating!r}")

    ts = time.time() if timestamp is None else timestamp
    answer_text = answer or ""
    record: dict[str, object] = {
        "timestamp": ts,
        "iso_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts)),
        "query": query,
        "answer_summary": answer_text[:_SUMMARY_LEN],
        "answer_hash": hashlib.sha256(answer_text.encode("utf-8")).hexdigest(),
        "rating": rating,
        "comment": comment,
    }

    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    with FEEDBACK_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    return record


__all__ = ["record_feedback", "FEEDBACK_FILE", "FEEDBACK_DIR", "Rating"]
