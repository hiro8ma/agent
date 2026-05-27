from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

# Allow running as `uv run python bin/web_researcher.py` without installing the package.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.web_researcher.runner import (  # noqa: E402
    DEFAULT_MAX_RESULTS,
    DEFAULT_WORKSPACE,
    run,
)


def _terminal_approval(payload: dict[str, Any]) -> bool:
    """Prompt the operator to approve a pending write_file in the terminal."""

    print("\n--- HITL approval required ---", file=sys.stderr)
    print(f"action : {payload.get('action')}", file=sys.stderr)
    print(f"path   : {payload.get('path')}", file=sys.stderr)
    print(f"bytes  : {payload.get('bytes')}", file=sys.stderr)
    preview = str(payload.get("preview", ""))
    print(f"preview:\n{preview}\n", file=sys.stderr)
    answer = input("approve write? [y/N] ").strip().lower()
    return answer in ("y", "yes")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="web_researcher",
        description=(
            "Research a topic on the web (Tavily) and save an HTML report. "
            "File writes pause for human approval (LangGraph interrupt)."
        ),
    )
    parser.add_argument("--topic", required=True, help="Research topic or question.")
    parser.add_argument(
        "--workspace",
        default=DEFAULT_WORKSPACE,
        help="Directory the report is written into (writes are confined here).",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=DEFAULT_MAX_RESULTS,
        help="Max web search results per query.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Auto-approve every write_file without prompting.",
    )
    args = parser.parse_args()

    if not os.environ.get("TAVILY_API_KEY"):
        print("error: TAVILY_API_KEY is not set. Export it before running.", file=sys.stderr)
        return 1
    if not os.environ.get("OPENAI_API_KEY"):
        print("error: OPENAI_API_KEY is not set. Export it before running.", file=sys.stderr)
        return 1

    approve = (lambda _payload: True) if args.yes else _terminal_approval

    try:
        answer = run(
            topic=args.topic,
            approve=approve,
            workspace=args.workspace,
            max_results=args.max_results,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(answer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
