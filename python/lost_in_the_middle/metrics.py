from __future__ import annotations

import re
import string
from collections.abc import Sequence

PUNCTUATION = set(string.punctuation)
ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize_answer(text: str) -> str:
    text = "".join(ch for ch in text.lower() if ch not in PUNCTUATION)
    return " ".join(ARTICLES.sub(" ", text).split())


def first_line(output: str) -> str:
    return output.strip().split("\n")[0].strip()


def is_correct(output: str, answers: Sequence[str]) -> bool:
    prediction = normalize_answer(first_line(output))
    return any(normalize_answer(answer) in prediction for answer in answers)
