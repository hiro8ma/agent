"""Compare three tool-selection strategies on the same query.

Runs native / semantic / hierarchical over one shared catalog of six fake tools and
prints, per strategy: the chosen tool, the args, how many tools the strategy considered,
and the LLM call count (the cost axis). Without an LLM key every strategy runs offline on
deterministic fakes; with OPENAI_API_KEY the real LLM and embeddings are used.

Run:
    uv run python bin/tool_selection.py --query "Solve 2x+3=7" --fake
    uv run python bin/tool_selection.py --query "notify the team in #general" --fake
    uv run python bin/tool_selection.py --strategy semantic --query "convert 10 km to miles"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.tool_selection import hierarchical, native, semantic  # noqa: E402
from agents.tool_selection.catalog import ALL_TOOLS  # noqa: E402
from agents.tool_selection.llm import Selection  # noqa: E402
from agents.tool_selection.semantic import select_matches  # noqa: E402

_SCALE_NOTE = {
    "native": "small, stable catalog",
    "semantic": "large catalog (top-k recall)",
    "hierarchical": "large categorized catalog",
}


def _run_one(strategy: str, query: str, top_k: int, use_fake: bool) -> Selection:
    if strategy == "native":
        return native.select(query, use_fake=use_fake)
    if strategy == "semantic":
        return semantic.select(query, top_k=top_k, use_fake=use_fake)
    if strategy == "hierarchical":
        return hierarchical.select(query, use_fake=use_fake)
    raise ValueError(f"unknown strategy: {strategy}")


def _print(sel: Selection, top_k: int, query: str, use_fake: bool) -> None:
    print(f"\n=== strategy: {sel.strategy} ({'fake' if sel.is_fake else 'real-llm'}) ===")
    print(f"  chosen tool : {sel.tool_name}")
    print(f"  args        : {sel.args}")
    print(f"  considered  : {sel.considered}")
    if sel.strategy == "semantic":
        matches = select_matches(query, top_k, use_fake=use_fake)
        ranked = ", ".join(f"{m.tool_name}={m.score:.3f}" for m in matches)
        print(f"  top-{top_k} cosine: {ranked}")
    print(f"  LLM calls   : {sel.llm_calls}  (cost axis)")
    print(f"  fits        : {_SCALE_NOTE[sel.strategy]}")
    print(f"  result      : {sel.result}")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="tool_selection",
        description=(
            "Compare native / semantic / hierarchical tool selection on one query over a "
            "shared 6-tool catalog. Shows the chosen tool and LLM call count per strategy. "
            "Runs offline with deterministic fakes when no OPENAI_API_KEY is set."
        ),
    )
    parser.add_argument("--query", required=True, help="The request to route to a tool.")
    parser.add_argument(
        "--strategy",
        choices=("native", "semantic", "hierarchical", "all"),
        default="all",
        help="Which strategy to run (default: all).",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Top-k for semantic (default 3).")
    parser.add_argument(
        "--fake", action="store_true", help="Force offline fakes even when a key is set."
    )
    args = parser.parse_args()

    use_fake = args.fake or not os.environ.get("OPENAI_API_KEY")
    if use_fake and not args.fake:
        print("note: no OPENAI_API_KEY found, using offline fakes.", file=sys.stderr)

    strategies = (
        ["native", "semantic", "hierarchical"] if args.strategy == "all" else [args.strategy]
    )

    print(f"catalog: {len(ALL_TOOLS)} tools | query: {args.query!r}")
    try:
        for strat in strategies:
            sel = _run_one(strat, args.query, args.top_k, use_fake)
            _print(sel, args.top_k, args.query, use_fake)
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.strategy == "all":
        print("\n=== cost summary (LLM calls) ===")
        print("  native=1  semantic=1(+1 embed)  hierarchical=2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
