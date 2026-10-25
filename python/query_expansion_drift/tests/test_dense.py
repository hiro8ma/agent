import numpy as np
import pytest

import dense
from metrics import rank

RNG = np.random.default_rng(0)
DOCS = dense.normalize(RNG.normal(size=(40, 8)))
QUERY = RNG.normal(size=(3, 8)) * 5.0
GENERATED = RNG.normal(size=(3, 8)) * 0.1

ALPHA_CASES = {
    "alpha 1 ranks like the original query": (1.0, QUERY),
    "alpha 0 ranks like the generated text": (0.0, GENERATED),
}


@pytest.mark.parametrize(("alpha", "reference"), ALPHA_CASES.values(), ids=ALPHA_CASES.keys())
def test_interpolation_endpoints_match_single_vector(alpha: float, reference: np.ndarray) -> None:
    vec = dense.interpolate(QUERY, GENERATED, alpha)

    got = rank(dense.scores(DOCS, vec), len(DOCS))
    want = rank(dense.scores(DOCS, reference), len(DOCS))

    np.testing.assert_array_equal(got, want)


def test_interpolation_normalizes_before_mixing() -> None:
    vec = dense.interpolate(QUERY, GENERATED, 0.5)

    want = 0.5 * dense.normalize(QUERY) + 0.5 * dense.normalize(GENERATED)
    np.testing.assert_allclose(vec, want)


def test_mean_with_query_weights_each_vector_equally() -> None:
    vec = dense.mean_with_query(QUERY, [GENERATED, GENERATED * 3.0])

    want = (dense.normalize(QUERY) + 2 * dense.normalize(GENERATED)) / 3
    np.testing.assert_allclose(vec, want)
