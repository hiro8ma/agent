"""A end-to-end RAG pipeline: chunk → embed → index → retrieve → augment → generate.

Knowledge vs memory (kept consistent with `retrieval.py`):

- *Knowledge* (RAG) is static external reference material — docs, a manual, a wiki —
  indexed once and queried read-only. RAG's whole job is to inject this knowledge into
  the prompt as *verifiable, cited facts* so the model answers from retrieved text
  instead of from its parametric guess. The corpus does not grow from a conversation.
- *Memory* is the agent's own history (notes, prior turns), written as it runs. That is
  `graph.py` / `retrieval.py`. This module is knowledge, not memory.

Two phases, mirroring every production RAG system:

1. *Indexing* (offline, once): split each document into overlapping chunks, embed them,
   and store them. `chunk_text` does the splitting; `MemoryStore` (reused) does the
   embed + index.
2. *Serving* (online, per query): embed the question, pull the top-k nearest chunks via
   semantic search, build an augmented prompt that pins those chunks under the question,
   and let the LLM answer grounded in them — with the source chunks shown so the answer
   is checkable.

Runs offline on the deterministic fake embedder and fake LLM when no key is set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agents.memory.retrieval import MemoryStore, semantic_search


@dataclass
class Chunk:
    """One indexed unit of knowledge plus the source metadata that makes it citable."""

    text: str
    source: str
    index: int  # position of this chunk within its source document


# Generic, fictional reference docs (a tiny "knowledge base"). No real entities.
DEFAULT_DOCS: dict[str, str] = {
    "handbook-deploys": (
        "Deployments run through a staged pipeline. A change is first promoted to the "
        "staging environment where the integration test suite runs end to end. Only after "
        "every staging check passes is the change eligible for production. Production "
        "rollout is gradual: traffic shifts to the new version in increments so a "
        "regression affects a small slice of users before a full cutover. A failed health "
        "check during rollout triggers an automatic rollback to the previous version."
    ),
    "handbook-onboarding": (
        "New contributors receive access on their first day through the central identity "
        "provider. Each person is added to exactly the groups their role requires, which "
        "keeps permissions least-privilege by default. Expense reimbursement is filed in "
        "the finance portal within thirty days of a purchase, and approvals are handled by "
        "the contributor's reporting lead. Equipment requests follow the same portal with a "
        "separate hardware approval step."
    ),
    "handbook-incidents": (
        "When an incident is declared, the on-call engineer becomes the incident "
        "commander and owns communication until handoff. Severity is set by user impact, "
        "not by the size of the broken component. The first action is to stop the bleeding "
        "by rolling back or disabling the faulty path, and only then to investigate root "
        "cause. Every incident closes with a blameless review whose action items are "
        "tracked to completion."
    ),
}


def chunk_text(text: str, window_size: int = 30, overlap: int = 8) -> list[str]:
    """Split text into overlapping fixed-size windows (tokens = whitespace words).

    Slides a `window_size`-word window forward by `window_size - overlap` each step.
    The overlap matters: a sentence whose meaning spans a chunk boundary would be cut in
    half by non-overlapping windows, so neither chunk would carry the full idea and a
    query about it could miss. Repeating `overlap` words into the next window keeps the
    boundary idea intact in at least one retrievable chunk.
    """

    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if not 0 <= overlap < window_size:
        raise ValueError("overlap must satisfy 0 <= overlap < window_size")

    words = text.split()
    if not words:
        return []

    step = window_size - overlap
    chunks: list[str] = []
    for start in range(0, len(words), step):
        window = words[start : start + window_size]
        chunks.append(" ".join(window))
        if start + window_size >= len(words):
            break  # last window already reached the end; avoid trailing duplicates
    return chunks


def build_chunks(
    docs: dict[str, str],
    window_size: int = 30,
    overlap: int = 8,
) -> list[Chunk]:
    """Indexing phase: chunk every document, carrying its source for later citation."""

    chunks: list[Chunk] = []
    for source, text in docs.items():
        for i, piece in enumerate(chunk_text(text, window_size, overlap)):
            chunks.append(Chunk(text=piece, source=source, index=i))
    return chunks


@dataclass
class RagIndex:
    """An indexed knowledge base: chunks plus the embedding store over them."""

    chunks: list[Chunk]
    store: MemoryStore

    @classmethod
    def build(
        cls,
        docs: dict[str, str],
        window_size: int = 30,
        overlap: int = 8,
    ) -> RagIndex:
        chunks = build_chunks(docs, window_size, overlap)
        store = MemoryStore(documents=[c.text for c in chunks])
        return cls(chunks=chunks, store=store)


@dataclass
class Retrieved:
    """A retrieved chunk with its similarity score, kept for display and citation."""

    chunk: Chunk
    score: float


def retrieve(index: RagIndex, query: str, top_k: int = 3, use_fake: bool = False) -> list[Retrieved]:
    """Serving phase, step 1: pull the top-k chunks nearest the question."""

    hits = semantic_search(index.store, query, top_k=top_k, use_fake=use_fake)
    return [Retrieved(chunk=index.chunks[i], score=score) for i, score in hits]


def build_augmented_prompt(query: str, retrieved: list[Retrieved]) -> str:
    """Serving phase, step 2: pin retrieved chunks under the question as cited context.

    This is the core RAG move: the model is told to answer *from* the context block, so
    the facts are injected and checkable rather than recalled from parameters.
    """

    context = "\n".join(
        f"[{r.chunk.source}#{r.chunk.index}] {r.chunk.text}" for r in retrieved
    )
    return (
        "Answer the question using only the context below. Cite the [source#n] tags you "
        "rely on. If the context does not contain the answer, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}"
    )


class FakeRagModel(BaseChatModel):
    """No-network model that answers strictly from the injected context block.

    It extracts the first sentence of the top retrieved chunk and echoes it with its
    citation tag. That is enough to demonstrate grounding: the answer is provably built
    from retrieved text, not from parametric memory, which is the whole point of RAG.
    """

    @property
    def _llm_type(self) -> str:
        return "fake-rag"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = messages[-1].content if messages else ""
        text = prompt if isinstance(prompt, str) else ""

        reply = "The context does not contain the answer."
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("[") and "]" in stripped:
                tag, _, body = stripped.partition("] ")
                tag = tag + "]"
                first_sentence = body.split(". ", 1)[0].rstrip(".")
                reply = f"{first_sentence}. (per {tag})"
                break

        message = AIMessage(content=reply)
        return ChatResult(generations=[ChatGeneration(message=message)])


def select_model(use_fake: bool) -> BaseChatModel:
    """Real provider when a key is present, deterministic fake otherwise (offline)."""

    if use_fake:
        return FakeRagModel()
    from core.providers.factory import select_provider

    return select_provider()


def generate(prompt: str, use_fake: bool = False) -> str:
    """Serving phase, step 3: generate an answer grounded in the augmented prompt."""

    model = select_model(use_fake)
    result = model.invoke([HumanMessage(content=prompt)])
    content = result.content
    return content if isinstance(content, str) else str(content)


@dataclass
class RagAnswer:
    """The full serving result: what was retrieved, the prompt sent, and the answer."""

    query: str
    retrieved: list[Retrieved]
    prompt: str
    answer: str


def answer_query(index: RagIndex, query: str, top_k: int = 3, use_fake: bool = False) -> RagAnswer:
    """Run the serving phase end to end: retrieve → augment → generate."""

    retrieved = retrieve(index, query, top_k=top_k, use_fake=use_fake)
    prompt = build_augmented_prompt(query, retrieved)
    answer = generate(prompt, use_fake=use_fake)
    return RagAnswer(query=query, retrieved=retrieved, prompt=prompt, answer=answer)
