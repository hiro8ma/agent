from __future__ import annotations

import pytest

from data import Document, parse_example


def ctx(title: str, isgold: bool) -> dict[str, object]:
    return {"title": title, "text": f"text of {title}", "isgold": isgold, "hasanswer": isgold}


def test_parse_example_splits_gold_and_distractors_in_order() -> None:
    row = {
        "question": "q",
        "answers": ["a", "b"],
        "ctxs": [ctx("G", True), ctx("D1", False), ctx("D2", False)],
    }

    example = parse_example(row)

    assert example.answers == ("a", "b")
    assert example.gold == Document(title="G", text="text of G")
    assert [d.title for d in example.distractors] == ["D1", "D2"]


def test_parse_example_rejects_a_row_whose_gold_is_not_first() -> None:
    row = {"question": "q", "answers": ["a"], "ctxs": [ctx("D1", False), ctx("G", True)]}

    with pytest.raises(ValueError):
        parse_example(row)
