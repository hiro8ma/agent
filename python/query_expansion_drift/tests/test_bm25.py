import numpy as np
import pytest

from bm25 import BM25, expand, tokenize

DOCS = [["apple", "pie"], ["banana", "split"], ["cherry", "tart"]]
QUERY = ["apple"]
GENERATED = ["banana", "banana"]

REPEAT_CASES = {
    "repeat 1 lets the generated terms win": (1, 1),
    "repeat 3 lets the original term win": (3, 0),
    "repeat 5 keeps the original term on top": (5, 0),
}


@pytest.mark.parametrize(("repeat", "want_top"), REPEAT_CASES.values(), ids=REPEAT_CASES.keys())
def test_repeating_query_scales_original_term_weight(repeat: int, want_top: int) -> None:
    index = BM25(DOCS)
    query_only = index.scores([QUERY])[0]
    generated_only = index.scores([GENERATED])[0]

    mixed = index.scores([expand(QUERY, GENERATED, repeat)])[0]

    np.testing.assert_allclose(mixed, repeat * query_only + generated_only)
    assert int(np.argmax(mixed)) == want_top


def test_terms_outside_vocabulary_are_ignored() -> None:
    index = BM25(DOCS)

    scores = index.scores([["durian"], ["durian", "apple"]])

    np.testing.assert_allclose(scores[0], 0.0)
    np.testing.assert_allclose(scores[1], index.scores([["apple"]])[0])


def test_tokenize_lowercases_and_drops_stopwords() -> None:
    assert tokenize("The IL-6 levels in Mice") == ["il", "6", "levels", "mice"]
