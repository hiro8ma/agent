"""Run a full RAG pipeline: index overlapping chunks, then retrieve + augment + generate.

Indexing phase (once): each reference doc is split into sliding-window chunks with
overlap and embedded into a store. Serving phase (per query): the question pulls the
top-k nearest chunks, those chunks are pinned into an augmented prompt as cited facts,
and the model answers grounded in them — knowledge injected as verifiable text, not
recalled from parameters.

Runs offline on the deterministic fake embedder + fake model when no OPENAI_API_KEY is
set.

Run:
    uv run python bin/rag_pipeline.py
    uv run python bin/rag_pipeline.py --question "what happens when a deploy fails health checks"
    uv run python bin/rag_pipeline.py --show-overlap
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.memory.rag_pipeline import (  # noqa: E402
    DEFAULT_DOCS,
    RagIndex,
    answer_query,
    chunk_text,
)

DEFAULT_QUESTIONS = [
    "what happens when a deploy fails its health checks during rollout",
    "how do new contributors get their expenses paid back",
    "who runs communication during an incident",
]


def _show_overlap() -> None:
    """Show that an overlapping split repeats boundary words across adjacent chunks."""

    sample = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
    no_overlap = chunk_text(sample, window_size=4, overlap=0)
    with_overlap = chunk_text(sample, window_size=4, overlap=2)
    print("sample:", sample)
    print("\nwindow=4 overlap=0 (boundary words appear once):")
    for c in no_overlap:
        print(f"  | {c}")
    print("\nwindow=4 overlap=2 (boundary words repeat into the next chunk):")
    for c in with_overlap:
        print(f"  | {c}")
    print(
        "\nNote: with overlap, 'gamma delta' / 'eta theta' etc. land in two chunks, so a "
        "query about a boundary phrase still hits a chunk that holds the whole phrase."
    )


def _run(index: RagIndex, question: str, top_k: int, use_fake: bool) -> None:
    result = answer_query(index, question, top_k=top_k, use_fake=use_fake)
    print(f"\nquestion: {question}")
    print("retrieved chunks (the injected, citable knowledge):")
    for r in result.retrieved:
        print(f"  [{r.score:5.3f}] [{r.chunk.source}#{r.chunk.index}] {r.chunk.text}")
    print(f"\nanswer:\n  {result.answer}")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="rag_pipeline",
        description="Index overlapping chunks, then retrieve + augment + generate (RAG).",
    )
    parser.add_argument("--question", help="Custom question (default: run the demo set).")
    parser.add_argument("--top-k", type=int, default=2, help="Chunks to retrieve per query.")
    parser.add_argument("--window-size", type=int, default=30, help="Chunk size in words.")
    parser.add_argument("--overlap", type=int, default=8, help="Words shared between chunks.")
    parser.add_argument(
        "--show-overlap",
        action="store_true",
        help="Demonstrate boundary-word overlap and exit.",
    )
    parser.add_argument(
        "--fake",
        action="store_true",
        help="Force the offline fake embedder and model even when a key is set.",
    )
    args = parser.parse_args()

    if args.show_overlap:
        _show_overlap()
        return 0

    use_fake = args.fake or not os.environ.get("OPENAI_API_KEY")
    if use_fake and not args.fake:
        print("note: no OPENAI_API_KEY found, using offline fake embedder + model.", file=sys.stderr)

    index = RagIndex.build(DEFAULT_DOCS, window_size=args.window_size, overlap=args.overlap)
    print(
        f"indexed {len(index.chunks)} chunks from {len(DEFAULT_DOCS)} docs "
        f"(window={args.window_size}, overlap={args.overlap})."
    )

    questions = [args.question] if args.question else DEFAULT_QUESTIONS
    for question in questions:
        _run(index, question, top_k=args.top_k, use_fake=use_fake)

    return 0


if __name__ == "__main__":
    sys.exit(main())
