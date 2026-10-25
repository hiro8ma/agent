from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeVar

from data import Document

T = TypeVar("T")
QuestionPlacement = Literal["after", "both"]

ANSWER_FORMAT = "Answer with a short phrase on a single line."
INSTRUCTION = (
    "Write a high-quality answer for the given question using only the provided search results "
    f"(some of which might be irrelevant). {ANSWER_FORMAT}"
)


def place_gold(
    gold: Document, distractors: Sequence[Document], k: int, position: int
) -> list[Document]:
    if not 1 <= position <= k:
        raise ValueError(f"position は1から{k}の範囲: {position}")
    if len(distractors) < k - 1:
        raise ValueError(f"外れの文書が{k - 1}件に足りない: {len(distractors)}")
    docs = list(distractors[: k - 1])
    docs.insert(position - 1, gold)
    return docs


def reorder_ends(ranked: Sequence[T]) -> list[T]:
    return list(ranked[0::2]) + list(ranked[1::2])[::-1]


def reordered_position(rank: int, k: int) -> int:
    return (rank + 1) // 2 if rank % 2 == 1 else k - rank // 2 + 1


def format_documents(docs: Sequence[Document]) -> str:
    return "\n".join(
        f"Document [{i}](Title: {d.title}) {d.text}" for i, d in enumerate(docs, start=1)
    )


def build_qa_prompt(question: str, docs: Sequence[Document], placement: QuestionPlacement) -> str:
    if not docs:
        raise ValueError("文書が空")
    search_results = format_documents(docs)
    if placement == "after":
        return f"{INSTRUCTION}\n\n{search_results}\n\nQuestion: {question}\nAnswer:"
    return f"{INSTRUCTION}\n\nQuestion: {question}\n\n{search_results}\n\nQuestion: {question}\nAnswer:"


def build_closedbook_prompt(question: str) -> str:
    return f"{ANSWER_FORMAT}\n\nQuestion: {question}\nAnswer:"
