"""GraphRAG demo: triple extraction -> graph build -> multi-hop traversal -> contrast.

Shows why a knowledge graph answers relational, multi-hop questions that a flat vector
RAG cannot. Same facts go into both pipelines; only the retrieval model differs.

    uv run python bin/graphrag.py
    uv run python bin/graphrag.py --path NLP "Attention Is All You Need"
    uv run python bin/graphrag.py --subgraph BERT --hops 2

Runs fully offline: the graph and BFS are pure Python; the vector-RAG side falls back to
the deterministic fake embedder + model when no OPENAI_API_KEY is set.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.memory.graphrag import (  # noqa: E402
    default_graph,
    extract_triples,
    format_path,
    models_built_with_competitor,
    query_inverse,
    shortest_path,
    subgraph,
    triples_to_text,
)
from agents.memory.rag_pipeline import RagIndex, answer_query  # noqa: E402


def _demo_extraction() -> None:
    print("=" * 78)
    print("1. TRIPLE EXTRACTION (rule-based, the offline stand-in for LLM extraction)")
    print("=" * 78)
    text = (
        "NLP is a subset of Machine Learning. "
        "BERT is based on Transformer. "
        "BERT is built with TensorFlow. "
        "TensorFlow is developed by Google."
    )
    print(f"input text:\n  {text}\n")
    print("extracted triples (subject --predicate--> object):")
    for t in extract_triples(text):
        print(f"  ({t.subject}) --{t.predicate}--> ({t.obj})")


def _demo_shortest_path(start: str, goal: str) -> None:
    graph = default_graph()
    print("\n" + "=" * 78)
    print("2. MULTI-HOP shortestPath (BFS): how are two entities related?")
    print("=" * 78)
    path = shortest_path(graph, start, goal)
    print(f"\nquery: how are '{start}' and '{goal}' related?")
    if path is None:
        print("  no path found.")
    else:
        print(f"  path ({len(path)} hops): {format_path(path)}")
    print(
        "\n  WHY this beats vector RAG: the answer is a *chain* across separate facts. "
        "Cosine\n  similarity would fetch chunks near each endpoint and miss the bridge "
        "nodes between them."
    )


def _demo_subgraph(center: str, hops: int) -> None:
    graph = default_graph()
    print("\n" + "=" * 78)
    print(f"3. N-HOP SUBGRAPH (local GraphRAG): everything within {hops} hop(s) of an entity")
    print("=" * 78)
    sub = subgraph(graph, center, hops=hops)
    print(f"\nquery (local): describe the neighbourhood of '{center}'")
    for t in sub:
        print(f"  ({t.subject}) --{t.predicate}--> ({t.obj})")


def _demo_multihop_question() -> None:
    graph = default_graph()
    print("\n" + "=" * 78)
    print("4. MULTI-HOP QUESTION: chain three edges no single chunk holds")
    print("=" * 78)
    print("\nquery: which models were built with a framework of the same TYPE as TensorFlow?")
    print("  (chain: TensorFlow --TYPE--> Framework <--TYPE-- PyTorch <--BUILT_WITH-- model)")
    for model, framework in models_built_with_competitor(graph, "TensorFlow"):
        print(f"  -> {model} (built with {framework})")

    print("\nquery (inverse one-hop): which models are an application of NLP?")
    for model in sorted(query_inverse(graph, "APPLICATION_OF", "NLP")):
        print(f"  -> {model}")


def _demo_contrast(use_fake: bool) -> None:
    graph = default_graph()
    print("\n" + "=" * 78)
    print("5. CONTRAST: flat vector RAG vs GraphRAG on the SAME knowledge")
    print("=" * 78)

    sentences = triples_to_text(graph.triples)
    docs = {f"fact-{i}": s for i, s in enumerate(sentences)}
    index = RagIndex.build(docs, window_size=12, overlap=0)

    question = "how are NLP and Deep Learning related?"
    print(f"\nquery (global/structural): {question}\n")

    print("  (a) flat vector RAG — retrieves k nearest chunks, no notion of a chain:")
    rag = answer_query(index, question, top_k=3, use_fake=use_fake)
    for r in rag.retrieved:
        print(f"      [{r.score:5.3f}] {r.chunk.text}")
    print(f"      answer: {rag.answer}")
    print(
        "      -> returns lexically near facts but cannot state the relationship; the "
        "linking\n         path is split across chunks."
    )

    print("\n  (b) GraphRAG — traverses the edge chain explicitly:")
    path = shortest_path(graph, "NLP", "Deep Learning")
    print(f"      path: {format_path(path or [])}")
    print("      answer: NLP USES Deep Learning directly (one hop); both are subsets of ML.")

    print("\nsummary:")
    print("  - flat RAG wins when the answer is inside one chunk (span retrieval).")
    print("  - GraphRAG wins on relational / multi-hop / structural questions (traversal).")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="graphrag",
        description="Triple extraction -> graph -> multi-hop traversal -> vs vector RAG.",
    )
    parser.add_argument("--path", nargs=2, metavar=("START", "GOAL"), help="shortestPath between two entities.")
    parser.add_argument("--subgraph", metavar="ENTITY", help="Extract the N-hop subgraph around an entity.")
    parser.add_argument("--hops", type=int, default=2, help="Hop radius for --subgraph (default 2).")
    parser.add_argument("--fake", action="store_true", help="Force the offline fake embedder + model.")
    args = parser.parse_args()

    use_fake = args.fake or not os.environ.get("OPENAI_API_KEY")

    if args.path:
        _demo_shortest_path(args.path[0], args.path[1])
        return 0
    if args.subgraph:
        _demo_subgraph(args.subgraph, args.hops)
        return 0

    if use_fake:
        print("note: no OPENAI_API_KEY found, using offline fake embedder + model.", file=sys.stderr)

    _demo_extraction()
    _demo_shortest_path("NLP", "Attention Is All You Need")
    _demo_subgraph("BERT", hops=2)
    _demo_multihop_question()
    _demo_contrast(use_fake)
    return 0


if __name__ == "__main__":
    sys.exit(main())
