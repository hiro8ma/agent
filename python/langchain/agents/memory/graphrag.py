"""GraphRAG: answer relational, multi-hop questions a flat vector RAG cannot.

Flat vector RAG (`rag_pipeline.py`) retrieves the top-k chunks nearest a query and pins
them into the prompt. That works when the answer lives *inside one chunk*. It breaks on
two shapes of question:

- *Multi-hop*: "what models were built with the tool that TensorFlow competes with?"
  needs to chain facts that live in different chunks. Cosine similarity has no notion of
  a chain; it just returns the k chunks most lexically/semantically near the query, and
  the intermediate hop ("the tool TensorFlow competes with") may not be near either end.
- *Global / structural*: "how are NLP and deep learning related?" asks about the *shape*
  of the knowledge, not a span of text. No single chunk states the relationship.

GraphRAG fixes this by representing knowledge as a graph of (subject, predicate, object)
RDF triples instead of opaque text chunks. A multi-hop question becomes a *graph
traversal* — BFS over edges — so the chain is followed explicitly rather than hoped for
in an embedding neighbourhood. shortestPath finds how two entities relate; an N-hop
subgraph gathers the local neighbourhood for grounding.

Runs offline with zero dependencies: the graph is a pure-Python adjacency list and the
traversal is pure-Python BFS. No Neo4j, no networkx, no embeddings, no credentials.
"""

from __future__ import annotations

import re
from collections import defaultdict, deque
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Triple:
    """One fact as an RDF triple: subject --predicate--> object.

    A triple is the atomic unit of a knowledge graph. Unlike a text chunk, a triple is
    typed and joinable: the object of one triple can be the subject of another, which is
    exactly what lets a query *chain* across facts (multi-hop) instead of matching a span.
    """

    subject: str
    predicate: str
    obj: str


# Generic AI-domain knowledge as triples. Only well-known public technical facts.
# The shape matters more than the contents: note that objects reappear as subjects
# (e.g. "Deep Learning" is both an object of NLP's edge and a subject elsewhere), which
# is what creates multi-hop paths to traverse.
DEFAULT_TRIPLES: list[Triple] = [
    Triple("NLP", "SUBSET_OF", "Machine Learning"),
    Triple("Computer Vision", "SUBSET_OF", "Machine Learning"),
    Triple("Deep Learning", "SUBSET_OF", "Machine Learning"),
    Triple("Machine Learning", "SUBSET_OF", "AI"),
    Triple("NLP", "USES", "Deep Learning"),
    Triple("Computer Vision", "USES", "Deep Learning"),
    Triple("Transformer", "IMPLEMENTS", "Deep Learning"),
    Triple("BERT", "BASED_ON", "Transformer"),
    Triple("GPT", "BASED_ON", "Transformer"),
    Triple("BERT", "APPLICATION_OF", "NLP"),
    Triple("GPT", "APPLICATION_OF", "NLP"),
    Triple("BERT", "BUILT_WITH", "TensorFlow"),
    Triple("GPT", "BUILT_WITH", "PyTorch"),
    Triple("TensorFlow", "DEVELOPED_BY", "Google"),
    Triple("PyTorch", "DEVELOPED_BY", "Meta"),
    Triple("TensorFlow", "TYPE", "Framework"),
    Triple("PyTorch", "TYPE", "Framework"),
    Triple("Transformer", "INTRODUCED_BY", "Attention Is All You Need"),
]


@dataclass
class KnowledgeGraph:
    """An in-memory directed knowledge graph backed by adjacency lists.

    Two adjacency maps are kept so traversal can go either direction:
    - `_out`: subject -> [(predicate, object)] for forward "what does X relate to" walks.
    - `_in`:  object  -> [(predicate, subject)] for backward "what relates to X" walks.

    BFS for shortestPath treats the graph as undirected (it may step along an edge in
    either direction), because relatedness is symmetric even when the relation is not:
    "BERT is built with TensorFlow" answers "how are BERT and TensorFlow related?".
    """

    triples: list[Triple] = field(default_factory=list)
    _out: dict[str, list[tuple[str, str]]] = field(default_factory=lambda: defaultdict(list), repr=False)
    _in: dict[str, list[tuple[str, str]]] = field(default_factory=lambda: defaultdict(list), repr=False)

    def __post_init__(self) -> None:
        for t in list(self.triples):
            self._index(t)

    def _index(self, t: Triple) -> None:
        self._out[t.subject].append((t.predicate, t.obj))
        self._in[t.obj].append((t.predicate, t.subject))

    def add(self, t: Triple) -> None:
        self.triples.append(t)
        self._index(t)

    def add_many(self, triples: list[Triple]) -> None:
        for t in triples:
            self.add(t)

    @property
    def entities(self) -> set[str]:
        return set(self._out) | set(self._in)

    def neighbors(self, entity: str) -> list[tuple[str, str, str]]:
        """All edges touching `entity`, as (predicate, other, direction).

        direction is "out" (entity is the subject) or "in" (entity is the object). This
        is the per-node step BFS uses, and what a subgraph extraction expands from.
        """

        out = [(p, o, "out") for p, o in self._out.get(entity, [])]
        inc = [(p, s, "in") for p, s in self._in.get(entity, [])]
        return out + inc


# --- Triple extraction -------------------------------------------------------------
#
# WHY rule-based here: production GraphRAG extracts triples with an LLM ("read this
# paragraph, emit (subject, predicate, object)"), which handles arbitrary phrasing and
# implicit relations. We use deterministic regex patterns instead so the demo runs
# offline and reproducibly. Limits of rule-based extraction, and why LLM extraction
# exists: it only fires on the exact surface forms below, cannot resolve coreference
# ("it", "the model"), cannot normalise synonyms ("uses" vs "is built on"), and cannot
# infer unstated relations. An LLM extractor trades determinism for that coverage.

_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^(.+?) is a subset of (.+)$", re.I), "SUBSET_OF"),
    (re.compile(r"^(.+?) is based on (.+)$", re.I), "BASED_ON"),
    (re.compile(r"^(.+?) is built with (.+)$", re.I), "BUILT_WITH"),
    (re.compile(r"^(.+?) is developed by (.+)$", re.I), "DEVELOPED_BY"),
    (re.compile(r"^(.+?) uses (.+)$", re.I), "USES"),
    (re.compile(r"^(.+?) implements (.+)$", re.I), "IMPLEMENTS"),
    (re.compile(r"^(.+?) is an application of (.+)$", re.I), "APPLICATION_OF"),
]


def extract_triples(text: str) -> list[Triple]:
    """Pull triples from one-relation-per-sentence text via surface patterns.

    Splits on sentence punctuation, then matches each clause against `_PATTERNS`. A real
    deployment would swap this for an LLM call; the function signature is the seam where
    that swap happens.
    """

    triples: list[Triple] = []
    for clause in re.split(r"[.\n;]+", text):
        clause = clause.strip().rstrip(".")
        if not clause:
            continue
        for pattern, predicate in _PATTERNS:
            m = pattern.match(clause)
            if m:
                subj = m.group(1).strip()
                obj = m.group(2).strip()
                triples.append(Triple(subj, predicate, obj))
                break
    return triples


# --- Multi-hop traversal (the core of GraphRAG) ------------------------------------


@dataclass
class PathStep:
    """One hop on a path.

    `frm`/`to` follow the *traversal* order (so a chain reads left to right), while
    `forward` records whether the underlying triple's direction matches that order. This
    keeps the rendered chain connected even when BFS walks an edge backward — relatedness
    is symmetric, so "Transformer IMPLEMENTS Deep Learning" can serve a walk arriving at
    Transformer from Deep Learning.
    """

    frm: str
    predicate: str
    to: str
    forward: bool


def shortest_path(graph: KnowledgeGraph, start: str, goal: str) -> list[PathStep] | None:
    """BFS shortest path between two entities, treating edges as undirected.

    WHY BFS over the graph instead of vector similarity: the question "how are A and B
    related?" has an answer only if a *chain* of facts connects them, and BFS returns the
    shortest such chain explicitly. A flat RAG would retrieve chunks near "A" and near
    "B" and never surface the bridge node in between. BFS is correct for an unweighted
    graph because the first time it reaches `goal` it has used the fewest hops.
    """

    if start not in graph.entities or goal not in graph.entities:
        return None
    if start == goal:
        return []

    visited = {start}
    queue: deque[list[PathStep]] = deque([[]])
    frontier: deque[str] = deque([start])
    while frontier:
        node = frontier.popleft()
        path = queue.popleft()
        for predicate, other, direction in graph.neighbors(node):
            if other in visited:
                continue
            step = PathStep(node, predicate, other, forward=direction == "out")
            new_path = [*path, step]
            if other == goal:
                return new_path
            visited.add(other)
            frontier.append(other)
            queue.append(new_path)
    return None


def subgraph(graph: KnowledgeGraph, center: str, hops: int = 1) -> list[Triple]:
    """Extract every triple within `hops` of `center` (the local-context retrieval).

    This is GraphRAG's analogue of vector top-k: instead of "k chunks nearest the query
    string", it returns "all facts within N hops of the query entity". The neighbourhood
    is structurally complete — no relevant edge of a reached node is missed — which is
    why local GraphRAG grounds relational answers that chunk retrieval would fragment.
    """

    if center not in graph.entities:
        return []

    reached = {center}
    frontier = {center}
    for _ in range(hops):
        nxt: set[str] = set()
        for node in frontier:
            for _predicate, other, _direction in graph.neighbors(node):
                if other not in reached:
                    nxt.add(other)
        reached |= nxt
        frontier = nxt
        if not frontier:
            break

    return [t for t in graph.triples if t.subject in reached and t.obj in reached]


def query_relation(graph: KnowledgeGraph, subject: str, predicate: str) -> list[str]:
    """Direct one-hop lookup: objects of `subject --predicate--> ?` edges."""

    return [o for p, o in graph._out.get(subject, []) if p == predicate]


def query_inverse(graph: KnowledgeGraph, predicate: str, obj: str) -> list[str]:
    """Inverse one-hop lookup: subjects of `? --predicate--> obj` edges.

    Powers questions like "which models were built with PyTorch?" — the object is fixed
    and we walk edges backward, which the `_in` adjacency makes O(deg) instead of a scan.
    """

    return [s for p, s in graph._in.get(obj, []) if p == predicate]


def models_built_with_competitor(graph: KnowledgeGraph, framework: str) -> list[tuple[str, str]]:
    """A genuine multi-hop answer: "models built with a framework of the same TYPE".

    Chains three edges no single chunk holds:
      framework --TYPE--> T  ;  other --TYPE--> T  ;  model --BUILT_WITH--> other.
    Returns (model, other_framework). This is the question class GraphRAG exists for.
    """

    kinds = query_relation(graph, framework, "TYPE")
    peers = {
        peer
        for kind in kinds
        for peer in query_inverse(graph, "TYPE", kind)
        if peer != framework
    }
    results: list[tuple[str, str]] = []
    for peer in sorted(peers):
        for model in sorted(query_inverse(graph, "BUILT_WITH", peer)):
            results.append((model, peer))
    return results


# --- Rendering helpers (used by the bin demo) --------------------------------------


def format_path(path: list[PathStep]) -> str:
    """Render a path as a readable chain: A --pred--> B --pred--> C."""

    if not path:
        return "(same entity)"
    parts = [path[0].frm]
    for step in path:
        arrow = f"--{step.predicate}-->" if step.forward else f"<--{step.predicate}--"
        parts.append(arrow)
        parts.append(step.to)
    return " ".join(parts)


def triples_to_text(triples: list[Triple]) -> list[str]:
    """Flatten triples into the kind of sentence a flat vector RAG would index.

    Used by the contrast demo to feed the *same* knowledge to the vector pipeline, so the
    comparison is apples-to-apples: same facts, different retrieval model.
    """

    verb = {
        "SUBSET_OF": "is a subset of",
        "USES": "uses",
        "IMPLEMENTS": "implements",
        "BASED_ON": "is based on",
        "APPLICATION_OF": "is an application of",
        "BUILT_WITH": "is built with",
        "DEVELOPED_BY": "is developed by",
        "TYPE": "is a",
        "INTRODUCED_BY": "was introduced by",
    }
    return [f"{t.subject} {verb.get(t.predicate, t.predicate)} {t.obj}." for t in triples]


def default_graph() -> KnowledgeGraph:
    return KnowledgeGraph(triples=list(DEFAULT_TRIPLES))
