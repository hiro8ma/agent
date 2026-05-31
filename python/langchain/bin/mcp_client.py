from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Allow running as `uv run python bin/mcp_client.py` without installing the package.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.mcp_client.runner import list_tools, run  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="mcp_client",
        description=(
            "Connect to the sibling mcp/ servers over stdio with "
            "MultiServerMCPClient, bind their tools, and answer a question."
        ),
    )
    parser.add_argument(
        "--question", help="Question to answer using MCP tools (needs OPENAI_API_KEY)."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the tools exposed by the MCP servers and exit (no LLM needed).",
    )
    parser.add_argument(
        "--mcp-repo",
        default=None,
        help="Path to the mcp/ repo. Defaults to MCP_REPO_PATH or sibling autodetect.",
    )
    args = parser.parse_args()

    try:
        if args.list:
            tools = asyncio.run(list_tools(args.mcp_repo))
            print("MCP tools:")
            for name, description in tools:
                first_line = description.strip().splitlines()[0] if description else ""
                print(f"  - {name}: {first_line}")
            return 0

        if not args.question:
            print("error: pass --question or --list.", file=sys.stderr)
            return 1

        if not os.environ.get("OPENAI_API_KEY"):
            print(
                "error: OPENAI_API_KEY is not set. Export it before running.",
                file=sys.stderr,
            )
            return 1

        print(asyncio.run(run(args.question, args.mcp_repo)))
        return 0
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
