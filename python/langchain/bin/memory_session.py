"""Demonstrate that a LangGraph checkpointer is what gives an agent cross-turn memory.

Runs the same two-turn conversation ("hi, I'm bob" -> "what's my name?") twice:
  1. through a checkpointer-less graph  -> the agent cannot recall the name
  2. through an InMemorySaver + thread_id -> the agent answers "bob"

Runs offline with a fake model when no LLM key is set.

Run:
    uv run python bin/memory_session.py
    uv run python bin/memory_session.py --fake
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.memory import reply_without_memory, run_session  # noqa: E402

TURNS = ["hi, I'm bob", "what's my name?"]


def _print_dialogue(title: str, replies: list[str]) -> None:
    print(f"\n=== {title} ===")
    for turn, reply in zip(TURNS, replies, strict=True):
        print(f"  user> {turn}")
        print(f"  bot > {reply}")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="memory_session",
        description="Show how a checkpointer turns a stateless chat graph into one with memory.",
    )
    parser.add_argument(
        "--fake",
        action="store_true",
        help="Force the offline fake model even when a key is set.",
    )
    parser.add_argument("--thread-id", default="bob-session", help="Checkpointer thread id.")
    args = parser.parse_args()

    use_fake = args.fake or not os.environ.get("OPENAI_API_KEY")
    if use_fake and not args.fake:
        print("note: no OPENAI_API_KEY found, using offline fake model.", file=sys.stderr)

    _print_dialogue("no checkpointer (amnesiac)", reply_without_memory(TURNS, use_fake=use_fake))
    _print_dialogue(
        f"InMemorySaver + thread_id={args.thread_id!r} (remembers)",
        run_session(TURNS, thread_id=args.thread_id, use_fake=use_fake),
    )

    print(
        "\nWhy: the node only sees the state it is handed. Without a saver each turn "
        "starts cold, so turn 2 never sees the introduction. The saver reloads the "
        "thread's state first, replaying turn 1 into turn 2."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
