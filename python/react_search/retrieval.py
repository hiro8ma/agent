from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

KENPOU_PATH = Path(__file__).resolve().parents[2] / "go" / "internal" / "enginecmp" / "kenpou.json"

_SPLIT = re.compile(r"[\s、。，．,.・「」『』（）()\[\]［］:：?？!！/]+")
_ARTICLE_NUMBER = re.compile(r"第([0-9]+|[一二三四五六七八九十百]+)条")
_KANJI_DIGITS = {c: i for i, c in enumerate("一二三四五六七八九", 1)}


@dataclass(frozen=True)
class Article:
    number: int
    text: str

    @property
    def label(self) -> str:
        return self.text.split("　", 1)[0]

    @property
    def body(self) -> str:
        parts = self.text.split("　", 1)
        return parts[1] if len(parts) == 2 else parts[0]

    def one_line(self) -> str:
        return self.text.replace("\n", "　")


def load_articles(path: Path = KENPOU_PATH) -> list[Article]:
    texts: list[str] = json.loads(path.read_text(encoding="utf-8"))
    return [Article(i + 1, t) for i, t in enumerate(texts)]


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def segments(text: str) -> list[str]:
    return [s for s in _SPLIT.split(normalize(text)) if s]


def bigrams(text: str) -> list[str]:
    out: list[str] = []
    for seg in segments(text):
        if len(seg) == 1:
            out.append(seg)
        else:
            out.extend(seg[i : i + 2] for i in range(len(seg) - 1))
    return out


def kanji_to_int(text: str) -> int:
    total, current = 0, 0
    for ch in text:
        if ch in _KANJI_DIGITS:
            current = _KANJI_DIGITS[ch]
        else:
            total += (current or 1) * (100 if ch == "百" else 10)
            current = 0
    return total + current


def article_numbers(text: str) -> set[int]:
    out = set()
    for m in _ARTICLE_NUMBER.finditer(normalize(text)):
        g = m.group(1)
        out.add(int(g) if g.isdigit() else kanji_to_int(g))
    return out


def containment(part: str, whole: str) -> float:
    p = set(bigrams(part))
    if not p:
        return 0.0
    return len(p & set(bigrams(whole))) / len(p)


class BM25:
    def __init__(self, articles: list[Article], k1: float = 1.2, b: float = 0.75) -> None:
        self.articles = articles
        self.k1 = k1
        self.b = b
        self.tfs = [Counter(bigrams(a.text)) for a in articles]
        self.lengths = [sum(tf.values()) for tf in self.tfs]
        self.avg_length = sum(self.lengths) / len(self.lengths)
        df = Counter(t for tf in self.tfs for t in tf)
        n = len(articles)
        self.idf = {t: math.log((n - c + 0.5) / (c + 0.5) + 1) for t, c in df.items()}

    def scores(self, query: str) -> list[float]:
        terms = set(bigrams(query))
        out = []
        for tf, length in zip(self.tfs, self.lengths, strict=True):
            norm = self.k1 * (1 - self.b + self.b * length / self.avg_length)
            out.append(
                sum(self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in terms if t in tf)
            )
        return out

    def search(self, query: str, k: int = 3) -> list[Article]:
        scored = [(s, a) for s, a in zip(self.scores(query), self.articles, strict=True) if s > 0]
        scored.sort(key=lambda x: (-x[0], x[1].number))
        return [a for _, a in scored[:k]]


def substring_search(articles: list[Article], query: str, k: int = 3) -> list[Article]:
    terms = set(segments(query))
    scored = []
    for a in articles:
        text = normalize(a.text)
        hits = sum(1 for t in terms if t in text)
        if hits:
            scored.append((hits, a))
    scored.sort(key=lambda x: (-x[0], x[1].number))
    return [a for _, a in scored[:k]]
