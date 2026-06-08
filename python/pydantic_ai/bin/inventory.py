from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow running as `uv run python bin/inventory.py` without installing the package.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.inventory import run  # noqa: E402
from agents.inventory.agent import _default_warehouse  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="inventory",
        description=(
            "Type-safe PydanticAI inventory agent. Reads stock via a tool, "
            "decides reorder, and returns a structured ReorderDecision. "
            "Runs offline with a FunctionModel when no LLM key is set."
        ),
    )
    parser.add_argument(
        "--query",
        default="widget-a",
        help=f"SKU to evaluate. Known: {', '.join(sorted(_default_warehouse()))}",
    )
    parser.add_argument(
        "--fake",
        action="store_true",
        help="Force the offline FunctionModel even when a key is set.",
    )
    args = parser.parse_args()

    use_fake = args.fake or not os.environ.get("OPENAI_API_KEY")
    if use_fake and not args.fake:
        print("note: no OPENAI_API_KEY found, using offline FunctionModel.\n", file=sys.stderr)

    decision, is_fake = run(args.query, use_fake=use_fake)

    print(f"=== ReorderDecision ({'offline' if is_fake else 'live LLM'}) ===")
    print(f"  item     : {decision.item}")
    print(f"  reorder  : {decision.reorder}")
    print(f"  quantity : {decision.quantity}")
    print(f"  reason   : {decision.reason}")
    print(f"\n  type     : {type(decision).__name__} (output_type で型保証)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
