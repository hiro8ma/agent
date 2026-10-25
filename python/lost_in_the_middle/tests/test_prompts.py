from __future__ import annotations

import pytest

from data import Document
from prompts import (
    ANSWER_FORMAT,
    QuestionPlacement,
    build_closedbook_prompt,
    build_qa_prompt,
    place_gold,
    reorder_ends,
    reordered_position,
)

GOLD = Document(title="Gold", text="The answer is here.")
DISTRACTORS = tuple(Document(title=f"D{i}", text=f"distractor {i}") for i in range(1, 20))


@pytest.mark.parametrize(
    ("ranked", "want"),
    [
        ([1, 2, 3, 4, 5, 6], [1, 3, 5, 6, 4, 2]),
        ([1], [1]),
        ([1, 2, 3, 4, 5], [1, 3, 5, 4, 2]),
        ([1, 2], [1, 2]),
        ([], []),
    ],
)
def test_reorder_ends_puts_odd_ranks_first_and_even_ranks_from_the_tail(
    ranked: list[int], want: list[int]
) -> None:
    assert reorder_ends(ranked) == want


@pytest.mark.parametrize("k", [1, 2, 5, 6, 20])
def test_reordered_position_matches_where_reorder_ends_puts_each_rank(k: int) -> None:
    reordered = reorder_ends(list(range(1, k + 1)))
    assert [reordered.index(r) + 1 for r in range(1, k + 1)] == [
        reordered_position(r, k) for r in range(1, k + 1)
    ]


@pytest.mark.parametrize(("k", "position"), [(20, 1), (20, 10), (20, 20), (5, 3), (1, 1)])
def test_place_gold_inserts_gold_and_keeps_distractor_order(k: int, position: int) -> None:
    docs = place_gold(GOLD, DISTRACTORS, k, position)

    assert len(docs) == k
    assert docs[position - 1] == GOLD
    assert [d for d in docs if d != GOLD] == list(DISTRACTORS[: k - 1])


@pytest.mark.parametrize(("k", "position"), [(20, 0), (20, 21), (21, 1)])
def test_place_gold_rejects_out_of_range(k: int, position: int) -> None:
    with pytest.raises(ValueError):
        place_gold(GOLD, DISTRACTORS, k, position)


def test_qa_prompt_after_puts_question_once_after_numbered_documents() -> None:
    prompt = build_qa_prompt("who wrote it", [GOLD, DISTRACTORS[0]], "after")

    assert prompt.count("Question: who wrote it") == 1
    assert prompt.index("Document [2](Title: D1) distractor 1") < prompt.index("Question:")
    assert "Document [1](Title: Gold) The answer is here.\nDocument [2]" in prompt
    assert prompt.endswith("Question: who wrote it\nAnswer:")


def test_qa_prompt_both_puts_question_before_and_after_documents() -> None:
    prompt = build_qa_prompt("who wrote it", [GOLD, DISTRACTORS[0]], "both")

    first = prompt.index("Question: who wrote it")
    last = prompt.rindex("Question: who wrote it")
    assert first < prompt.index("Document [1]") < prompt.index("Document [2]") < last
    assert prompt.count("Question: who wrote it") == 2
    assert prompt.endswith("Question: who wrote it\nAnswer:")


def test_qa_prompt_rejects_empty_documents() -> None:
    with pytest.raises(ValueError):
        build_qa_prompt("q", [], "after")


def test_closedbook_prompt_has_no_documents() -> None:
    prompt = build_closedbook_prompt("who wrote it")

    assert "Document" not in prompt
    assert prompt.startswith(ANSWER_FORMAT)
    assert prompt.endswith("\n\nQuestion: who wrote it\nAnswer:")


@pytest.mark.parametrize("placement", ["after", "both"])
def test_qa_prompt_asks_for_a_one_line_answer_before_the_documents(
    placement: QuestionPlacement,
) -> None:
    prompt = build_qa_prompt("who wrote it", [GOLD], placement)

    assert prompt.index(ANSWER_FORMAT) < prompt.index("Document [1]")
