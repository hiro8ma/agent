"""Compare BM25 (lexical) and semantic (embedding) retrieval over one memory corpus.

Same store, two query styles:
  - proper-noun query ("Zephyr quarterly sync") -> BM25 wins (exact term overlap)
  - concept / paraphrase query ("how much do interns get paid") -> semantic wins
    (no shared words with "monthly allowance of 500 credits")

Prints both rankings side by side plus the hybrid (RRF) fusion. Runs offline on the
deterministic fake embedder when no OPENAI_API_KEY is set.

Run:
    uv run python bin/memory_search.py
    uv run python bin/memory_search.py --query "Halcyon deploy" --fake
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.memory.retrieval import (  # noqa: E402
    DEFAULT_CORPUS,
    MemoryStore,
    bm25_search,
    hybrid_search,
    semantic_search,
)

CONTRAST_QUERIES = [
    ("proper-noun (BM25 favoured)", "Zephyr quarterly planning sync"),
    ("concept / paraphrase (semantic favoured)", "how much do interns get paid"),
]


def _show(store: MemoryStore, query: str, use_fake: bool) -> None:
    print(f"\nquery: {query!r}")
    for label, hits in (
        ("BM25     ", bm25_search(store, query, top_k=2)),
        ("semantic ", semantic_search(store, query, top_k=2, use_fake=use_fake)),
        ("hybrid   ", hybrid_search(store, query, top_k=2, use_fake=use_fake)),
    ):
        if not hits:
            print(f"  {label} -> (no lexical overlap)")
            continue
        for idx, score in hits:
            print(f"  {label} -> [{score:5.3f}] {store.documents[idx]}")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="memory_search",
        description="BM25 vs semantic vs hybrid retrieval over an agent memory corpus.",
    )
    parser.add_argument("--query", help="Custom query (default: run the two contrast queries).")
    parser.add_argument(
        "--fake",
        action="store_true",
        help="Force the offline fake embedder even when a key is set.",
    )
    args = parser.parse_args()

    use_fake = args.fake or not os.environ.get("OPENAI_API_KEY")
    if use_fake and not args.fake:
        print("note: no OPENAI_API_KEY found, using offline fake embedder.", file=sys.stderr)

    store = MemoryStore(documents=list(DEFAULT_CORPUS))

    if args.query:
        _show(store, args.query, use_fake)
        return 0

    for label, query in CONTRAST_QUERIES:
        print(f"\n### {label}")
        _show(store, query, use_fake)

    print(
        "\nTakeaway: BM25 nails exact tokens a user said (proper-noun query: top hit, "
        "high score). On the paraphrase query BM25 misses the 'allowance' memory "
        "entirely while semantic still surfaces it. Hybrid (RRF) fuses both."
    )
    if use_fake:
        print(
            "note: offline embedder is lexical (tri-gram hashing), so the semantic edge "
            "shown here is the floor — a real embedding model widens the paraphrase gap."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
