"""memory — session-spanning memory (checkpointer) and memory retrieval (BM25 vs semantic).

Two distinct ideas the agent world keeps separate:

- *Knowledge* (RAG): static external facts indexed once, queried read-only. The corpus
  does not change because the agent talked to a user. See `agents/doc_reader`.
- *Memory*: the agent's own conversation / event history, written as the agent runs and
  read back later. This package is about memory, not knowledge.

`graph.py`        — a one-node chat graph that shows why a LangGraph checkpointer is what
                    turns a stateless call into cross-turn (and cross-session) memory.
`retrieval.py`    — how to *search* a memory store: lexical BM25 vs semantic embeddings,
                    and why a production store blends both (hybrid).
`rag_pipeline.py` — knowledge, not memory: a full RAG pipeline (chunk → embed → index →
                    retrieve → augment → generate) over static reference docs.
`graphrag.py`     — knowledge as a graph of (subject, predicate, object) triples; answers
                    multi-hop / relational questions by BFS traversal, not chunk retrieval.
"""

from .graph import build_chat_graph, reply_without_memory, run_session
from .graphrag import (
    KnowledgeGraph,
    Triple,
    default_graph,
    extract_triples,
    shortest_path,
    subgraph,
)
from .rag_pipeline import RagAnswer, RagIndex, answer_query, chunk_text
from .retrieval import MemoryStore, bm25_search, semantic_search

__all__ = [
    "KnowledgeGraph",
    "MemoryStore",
    "RagAnswer",
    "RagIndex",
    "Triple",
    "answer_query",
    "bm25_search",
    "build_chat_graph",
    "chunk_text",
    "default_graph",
    "extract_triples",
    "reply_without_memory",
    "run_session",
    "semantic_search",
    "shortest_path",
    "subgraph",
]
