from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow running as `uv run python bin/reflexion.py` without installing the package.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.reflexion import DEFAULT_MAX_ATTEMPTS, run_reflexion  # noqa: E402


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in text.splitlines()) if text else prefix + "(none)"


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="reflexion",
        description=(
            "Reflexion-style self-improvement loop: act → evaluate → reflect → retry. "
            "Verbal reflections accumulate in an episodic buffer and are injected into "
            "the next attempt's prompt — nonparametric learning, no weight updates. "
            "Runs offline with a deterministic fake actor when no LLM key is set."
        ),
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help="Hard cap on attempts (loop termination guard).",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Use the real LLM actor (requires OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--memory-file",
        help="Persist reflections to this JSONL file (carries memory across runs).",
    )
    args = parser.parse_args()

    use_fake = not (args.llm and os.environ.get("OPENAI_API_KEY"))
    if args.llm and not os.environ.get("OPENAI_API_KEY"):
        print("note: --llm given but no OPENAI_API_KEY found, using offline fake actor.\n", file=sys.stderr)
    elif not args.llm:
        print("note: running with offline fake actor (pass --llm for a real model).\n", file=sys.stderr)

    result = run_reflexion(
        max_attempts=args.max_attempts,
        use_fake=use_fake,
        memory_file=args.memory_file,
    )

    for record in result.attempts:
        print(f"=== attempt {record.index} ===")
        print("[injected reflection]")
        print(_indent(record.injected_reflection))
        print("[code]")
        print(_indent(record.code))
        print(f"[result] {'PASS' if record.passed else 'FAIL'} "
              f"({record.passed_count}/{record.total} cases)")
        print("[observation]")
        print(_indent(record.observation))
        print()

    print("=== summary ===")
    print(f"actor: {'fake (offline)' if result.used_fake else 'llm'}")
    print(f"solved: {result.solved} after {len(result.attempts)} attempt(s)")
    print(f"reflections collected: {len(result.reflections)}")
    for i, r in enumerate(result.reflections, start=1):
        print(f"  {i}. {r}")

    return 0 if result.solved else 2


if __name__ == "__main__":
    sys.exit(main())
