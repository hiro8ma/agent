from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow running as `uv run python bin/reflection.py` without installing the package.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.reflection import draw_mermaid, stream  # noqa: E402
from agents.reflection.graph import DEFAULT_MAX_ITERATIONS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="reflection",
        description=(
            "Plan→Generate→Reflect loop on a LangGraph StateGraph. "
            "Streams each node update and stops at an iteration cap or a PASS verdict. "
            "Runs offline with a fake model when no LLM key is set."
        ),
    )
    parser.add_argument("--task", help="Task to plan, draft, and refine.")
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help="Hard cap on Generate passes (loop termination guard).",
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

    if not args.task:
        print("error: --task is required (or pass --mermaid).", file=sys.stderr)
        return 1

    use_fake = args.fake or not os.environ.get("OPENAI_API_KEY")
    if use_fake and not args.fake:
        print("note: no OPENAI_API_KEY found, using offline fake model.\n", file=sys.stderr)

    for node, update in stream(args.task, max_iterations=args.max_iterations, use_fake=use_fake):
        print(f"\n=== node: {node} ===")
        for key in ("plan", "draft", "critique"):
            if key in update:
                print(f"[{key}]\n{update[key]}")
        if "iteration" in update:
            print(f"[iteration] {update['iteration']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
