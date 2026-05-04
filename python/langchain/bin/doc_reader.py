from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow running as `uv run python bin/doc_reader.py` without installing the package.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.doc_reader.runner import (  # noqa: E402
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_CHUNKS,
    run,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="doc_reader",
        description="Answer a question over a single PDF using LangChain primitives.",
    )
    parser.add_argument("--pdf", required=True, help="Absolute path to the PDF file.")
    parser.add_argument("--question", required=True, help="Question to ask of the PDF.")
    parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Splitter chunk size."
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="Splitter chunk overlap.",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=DEFAULT_MAX_CHUNKS,
        help="Maximum chunks fed into the prompt context.",
    )
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "error: OPENAI_API_KEY is not set. Export it before running.",
            file=sys.stderr,
        )
        return 1

    pdf_path = os.path.abspath(args.pdf)

    try:
        answer = run(
            pdf_path=pdf_path,
            question=args.question,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            max_chunks=args.max_chunks,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(answer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
