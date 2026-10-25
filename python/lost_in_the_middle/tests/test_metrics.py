from __future__ import annotations

import pytest

from metrics import first_line, is_correct, normalize_answer


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("The Beatles", "beatles"),
        ("May 18, 2018.", "may 18 2018"),
        ("  An   apple  a day ", "apple day"),
        ("Röntgen", "röntgen"),
    ],
)
def test_normalize_answer_lowercases_and_drops_punctuation_articles_and_extra_spaces(
    text: str, want: str
) -> None:
    assert normalize_answer(text) == want


@pytest.mark.parametrize(
    ("output", "answers", "want"),
    [
        ("Based on the results, it was released on May 18, 2018.", ["May 18, 2018"], True),
        ("Wilhelm Conrad Röntgen won it.", ["Marie Curie", "Wilhelm Conrad Röntgen"], True),
        ("It was released in 2019.", ["May 18, 2018"], False),
        ("I am not sure.\nDocument [3] says May 18, 2018.", ["May 18, 2018"], False),
        ("\n\nMay 18, 2018\nmore text", ["May 18, 2018"], True),
        ("THE BEATLES!", ["the Beatles"], True),
    ],
)
def test_is_correct_checks_if_any_answer_appears_in_the_first_line(
    output: str, answers: list[str], want: bool
) -> None:
    assert is_correct(output, answers) is want


def test_first_line_skips_leading_blank_lines() -> None:
    assert first_line("\n  answer here  \nsecond") == "answer here"
