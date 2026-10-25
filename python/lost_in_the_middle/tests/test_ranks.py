from __future__ import annotations

from collections import Counter

import pytest

from ranks import allocate, assign_ranks, uniform_weights, zipf_weights


@pytest.mark.parametrize(
    ("n", "weights", "want"),
    [
        (40, uniform_weights(20), [2] * 20),
        (10, [1.0, 1.0, 1.0], [4, 3, 3]),
        (12, zipf_weights(3), [7, 3, 2]),
    ],
)
def test_allocate_splits_n_in_proportion_to_weights(
    n: int, weights: list[float], want: list[int]
) -> None:
    assert allocate(n, weights) == want


def test_assign_ranks_keeps_the_allocated_counts_and_is_reproducible() -> None:
    ranks = assign_ranks(50, zipf_weights(20), seed=3)

    counts = Counter(ranks)
    assert [counts[r] for r in range(1, 21)] == allocate(50, zipf_weights(20))
    assert ranks == assign_ranks(50, zipf_weights(20), seed=3)
    assert ranks != sorted(ranks)
