import math

import numpy as np
import pytest

from metrics import ndcg_at_k, overlap_at_k, rank, recall_at_k

RANKED = ["a", "b", "c", "d"]

NDCG_CASES = {
    "single relevant at rank 1": ({"a": 1}, 10, 1.0),
    "single relevant at rank 2": ({"b": 1}, 10, 1 / math.log2(3)),
    "two relevant at ranks 1 and 3": (
        {"a": 1, "c": 1},
        10,
        (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3)),
    ),
    "relevant outside the cutoff": ({"d": 1}, 3, 0.0),
    "unretrieved relevant raises the ideal": ({"a": 1, "z": 1}, 10, 1 / (1 + 1 / math.log2(3))),
    "graded relevance": ({"a": 1, "b": 2}, 10, (1 + 2 / math.log2(3)) / (2 + 1 / math.log2(3))),
}


@pytest.mark.parametrize(("qrels", "k", "want"), NDCG_CASES.values(), ids=NDCG_CASES.keys())
def test_ndcg_at_k(qrels: dict[str, int], k: int, want: float) -> None:
    assert ndcg_at_k(RANKED, qrels, k) == pytest.approx(want)


RECALL_CASES = {
    "all relevant retrieved": ({"a": 1, "d": 1}, 100, 1.0),
    "half retrieved": ({"b": 1, "z": 1}, 100, 0.5),
    "cutoff drops a relevant": ({"a": 1, "d": 1}, 2, 0.5),
    "zero-graded documents do not count": ({"a": 1, "z": 0}, 100, 1.0),
}


@pytest.mark.parametrize(("qrels", "k", "want"), RECALL_CASES.values(), ids=RECALL_CASES.keys())
def test_recall_at_k(qrels: dict[str, int], k: int, want: float) -> None:
    assert recall_at_k(RANKED, qrels, k) == pytest.approx(want)


def test_overlap_at_k_ignores_order() -> None:
    assert overlap_at_k(["a", "b", "c"], ["c", "x", "a"], 3) == pytest.approx(2 / 3)


def test_rank_breaks_ties_by_index() -> None:
    scores = np.array([[0.1, 0.5, 0.5, 0.2]])

    np.testing.assert_array_equal(rank(scores, 3), [[1, 2, 3]])
