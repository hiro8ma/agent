"""Semantic tool selection (tool-RAG) demo over the sibling mcp/ servers.

Standard MCP clients bind every tool to the LLM. Here we wire calc (7 tools) +
memory (4 tools) = 11 tools, then for each query embed the tools' ``name:
description``, embed the query, and keep only the top-k by cosine similarity.
The LLM (if present) sees k tools instead of 11.

What this proves:
    - math queries surface calc tools in the top-k
    - memory queries surface remember/recall in the top-k
    - the tool count handed to the LLM drops from 11 to k

Embeddings default to a key-free local FastEmbed (ONNX) model, so the selection
half runs with no API key. Pass --llm (with OPENAI_API_KEY) to actually invoke
the agent on the narrowed tool set.

Run:
    uv run python bin/semantic_tool_select.py             # selection only, key-free
    uv run python bin/semantic_tool_select.py --k 2
    uv run python bin/semantic_tool_select.py --llm       # also call the LLM (needs key)
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

from agents.mcp_client.runner import (  # noqa: E402
    run_semantic,
    select_tools_for_query,
)

DEMO_QUERIES: list[tuple[str, str]] = [
    ("math", "Add 12 and 8, then multiply the result by 3."),
    ("memory", "Remember that my favorite language is Go, then recall it later."),
    ("unrelated", "Write a haiku about the ocean at sunrise."),
]


async def _run(k: int, use_llm: bool, mcp_repo: str | None) -> int:
    servers = ["calc", "memory"]

    for label, query in DEMO_QUERIES:
        all_tools, matches = await select_tools_for_query(query, k, mcp_repo, servers)
        total = len(all_tools)

        print(f"\n=== [{label}] {query}")
        print(f"standard selection: bind all {total} tools")
        print(f"semantic selection: bind top-{len(matches)} tools (tool-RAG)")
        for rank, m in enumerate(matches, start=1):
            print(f"  {rank}. {m.tool.name:<14} cos={m.score:.3f}")
        reduction = total - len(matches)
        print(f"  -> handed {len(matches)}/{total} tools to the LLM (-{reduction})")

        if use_llm:
            answer = await run_semantic(query, k, mcp_repo, servers)
            print(f"  LLM answer: {answer}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="semantic_tool_select",
        description=(
            "tool-RAG demo: embed mcp/ tool descriptions, pick top-k by cosine "
            "similarity per query, and bind only those instead of all 11 tools."
        ),
    )
    parser.add_argument(
        "--k", type=int, default=3, help="Number of tools to keep per query (default 3)."
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Also invoke the agent on the narrowed tool set (needs OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--mcp-repo",
        default=None,
        help="Path to the mcp/ repo. Defaults to MCP_REPO_PATH or sibling autodetect.",
    )
    args = parser.parse_args()

    if args.llm and not os.environ.get("OPENAI_API_KEY"):
        print(
            "error: --llm needs OPENAI_API_KEY. Omit --llm for the key-free path.",
            file=sys.stderr,
        )
        return 1

    try:
        return asyncio.run(_run(args.k, args.llm, args.mcp_repo))
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
