from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow running as `uv run python bin/supervisor.py` without installing the package.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.supervisor import draw_mermaid, stream  # noqa: E402
from agents.supervisor.graph import DEFAULT_TOOL_ROUNDS  # noqa: E402

# Demo queries that exercise all three routes (fictional supply network).
DEMO_QUERIES = [
    "SKU-4471 just hit a stockout. What's the reorder strategy and demand forecast?",
    "Shipment SHP-908 is delayed. Where is it and can we reroute it?",
    "Score supplier SUP-12 and tell me if it is compliant.",
]


def _run_one(query: str, max_tool_rounds: int, use_fake: bool) -> None:
    print(f"\n########## query: {query}")
    final_answer = ""
    for node, update in stream(query, max_tool_rounds=max_tool_rounds, use_fake=use_fake):
        print(f"\n=== node: {node} ===")
        op = update.get("operation", {})
        if "route" in op:
            print(f"[route] supervisor -> {op['route']}")
        for message in update.get("messages", []):
            tool_calls = getattr(message, "tool_calls", None) or []
            if tool_calls:
                for call in tool_calls:
                    print(f"[tool_call] {call['name']}({call['args']})")
            elif message.type == "tool":
                print(f"[tool_result] {message.content}")
            elif message.content:
                print(f"[{message.type}] {message.content}")
        if "final_answer" in op and op["final_answer"]:
            final_answer = op["final_answer"]
    if final_answer:
        print("\n=== final answer ===")
        print(final_answer)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="supervisor",
        description=(
            "Supervisor-style multi-agent system on a LangGraph StateGraph. A supervisor "
            "routes a supply-chain request to one of three specialists (inventory / "
            "transportation / supplier), each bound to its own tool group; the specialist "
            "runs a tool-calling loop and answers. Runs offline with a fake model and fake "
            "tools when no LLM key is set."
        ),
    )
    parser.add_argument("--query", help="Supply-chain request to route.")
    parser.add_argument(
        "--max-tool-rounds",
        type=int,
        default=DEFAULT_TOOL_ROUNDS,
        help="Per-specialist tool-call loop cap (loop termination guard).",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the three built-in demo queries (one per specialist).",
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

    if not args.query and not args.demo:
        print("error: pass --query, --demo, or --mermaid.", file=sys.stderr)
        return 1

    use_fake = args.fake or not os.environ.get("OPENAI_API_KEY")
    if use_fake and not args.fake:
        print("note: no OPENAI_API_KEY found, using offline fake model.\n", file=sys.stderr)

    queries = DEMO_QUERIES if args.demo else [args.query]
    for query in queries:
        _run_one(query, args.max_tool_rounds, use_fake)

    return 0


if __name__ == "__main__":
    sys.exit(main())
