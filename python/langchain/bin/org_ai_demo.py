"""Organizational AI demo: many specialist agents, one shared org memory.

Picture: several agents share a single "organizational memory" exposed by the
mcp/memory server. The minimal loop is reference + write-back:

    Agent A  --remember-->  [ org memory (mcp/memory, SQLite+FTS5) ]  <--recall--  Agent B
                                       ^                                              |
                                       +--------------- remember (write-back) --------+

Each step below opens a *fresh* MultiServerMCPClient to mimic an independent
agent context. State survives only because it lives in the shared memory server,
not in any single agent. The LLM step is optional: with no API key the demo
still proves stdio connect + remember/recall round-trips.

Run:
    uv run python bin/org_ai_demo.py          # key-free loop (no LLM)
    uv run python bin/org_ai_demo.py --llm    # add an LLM recall step (needs OPENAI_API_KEY)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.mcp_client.runner import call_tool, list_tools, run  # noqa: E402

MEMORY = ["memory"]


async def _demo(mcp_repo: str | None, use_llm: bool) -> int:
    print("== org-AI memory loop (mcp/memory over stdio) ==\n")

    print("[0] connect + list memory tools")
    for name, desc in await list_tools(mcp_repo, MEMORY):
        first = desc.strip().splitlines()[0] if desc else ""
        print(f"    - {name}: {first}")
    print()

    fact = (
        "Org rule: production deploys run `make deploy`, then post to #ops on Slack."
    )
    print("[a] Agent-A remembers a decision into shared org memory")
    print("    ->", await call_tool(
        "remember",
        {"content": fact, "category": "decision", "tags": ["ops", "deploy"]},
        mcp_repo,
        MEMORY,
    ))
    print()

    print("[b] Agent-B (fresh client context) recalls it from shared memory")
    recalled = await call_tool(
        "recall",
        {"query": "how do we deploy to production", "limit": 3, "rerank": False},
        mcp_repo,
        MEMORY,
    )
    print("   ", recalled.replace("\n", "\n    "))
    print()

    if use_llm and os.environ.get("OPENAI_API_KEY"):
        print("[b'] LLM-backed agent answers using the recall tool")
        answer = await run(
            "Using memory, what is our production deploy procedure?",
            mcp_repo,
            MEMORY,
        )
        print("   ", answer.replace("\n", "\n    "))
        print()
    elif use_llm:
        print("[b'] skipped: OPENAI_API_KEY not set\n")

    print("[c] Agent-B writes the execution result back to shared memory")
    outcome = (
        "Executed deploy procedure on 2026-06-02: `make deploy` succeeded, "
        "#ops notified. No incidents."
    )
    print("    ->", await call_tool(
        "remember",
        {"content": outcome, "category": "experience", "tags": ["ops", "deploy", "result"]},
        mcp_repo,
        MEMORY,
    ))
    print()

    print("[d] stats — both the decision and its execution result now persist")
    print("   ", (await call_tool("memory_stats", {}, mcp_repo, MEMORY)).replace("\n", "\n    "))
    print("\n== loop complete: reference (recall) + write-back (remember) ==")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="org_ai_demo",
        description="Demo the reference + write-back loop over the shared org memory.",
    )
    parser.add_argument("--mcp-repo", default=None, help="Path to the mcp/ repo.")
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Add an LLM-backed recall step (needs OPENAI_API_KEY).",
    )
    args = parser.parse_args()
    try:
        return asyncio.run(_demo(args.mcp_repo, args.llm))
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
