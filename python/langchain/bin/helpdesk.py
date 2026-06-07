from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow running as `uv run python bin/helpdesk.py` without installing the package.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.helpdesk import draw_mermaid, stream  # noqa: E402
from agents.helpdesk.graph import DEFAULT_MAX_RETRIES  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="helpdesk",
        description=(
            "Plan-and-Execute helpdesk agent on a LangGraph StateGraph with hybrid RAG "
            "(keyword + vector retrieval). Splits an inquiry into independent subtasks, "
            "routes each to a retrieval tool, reflects, and synthesizes a final answer. "
            "Runs offline with a fake model and in-memory retrievers when no LLM key is set."
        ),
    )
    parser.add_argument("--query", help="Helpdesk inquiry to resolve.")
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Per-subtask retry cap after a failing reflection (loop termination guard).",
    )
    parser.add_argument(
        "--fake",
        action="store_true",
        help="Force the offline fake model even when a key is set.",
    )
    parser.add_argument(
        "--mermaid",
        action="store_true",
        help="Print the graph structure as mermaid text and exit.",
    )
    args = parser.parse_args()

    if args.mermaid:
        print(draw_mermaid())
        return 0

    if not args.query:
        print("error: --query is required (or pass --mermaid).", file=sys.stderr)
        return 1

    use_fake = args.fake or not os.environ.get("OPENAI_API_KEY")
    if use_fake and not args.fake:
        print("note: no OPENAI_API_KEY found, using offline fake model.\n", file=sys.stderr)

    final_answer = ""
    for node, update in stream(args.query, max_retries=args.max_retries, use_fake=use_fake):
        print(f"\n=== node: {node} ===")
        if "subtasks" in update:
            for i, sub in enumerate(update["subtasks"]):
                print(f"  subtask[{i}] {sub}")
        for key in ("draft", "critique"):
            if key in update:
                print(f"[{key}]\n{update[key]}")
        if "sub_answers" in update:
            print(f"[committed]\n{update['sub_answers'][-1]}")
        if "iteration" in update:
            print(f"[iteration] {update['iteration']}")
        if "final_answer" in update:
            final_answer = update["final_answer"]

    if final_answer:
        print("\n=== final answer ===")
        print(final_answer)

    return 0


if __name__ == "__main__":
    sys.exit(main())
