from __future__ import annotations

import re
from collections import Counter

import numpy as np
from scipy import sparse

# Lucene の EnglishAnalyzer の既定の停止語
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "if",
        "in",
        "into",
        "is",
        "it",
        "no",
        "not",
        "of",
        "on",
        "or",
        "such",
        "that",
        "the",
        "their",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "will",
        "with",
    ]
)
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


def expand(query: list[str], generated: list[str], repeat: int) -> list[str]:
    return query * repeat + generated


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 0.9, b: float = 0.4) -> None:
        self.vocab: dict[str, int] = {}
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        for i, toks in enumerate(docs):
            for term, count in Counter(toks).items():
                rows.append(i)
                cols.append(self.vocab.setdefault(term, len(self.vocab)))
                vals.append(count)
        n_docs = len(docs)
        tf = np.asarray(vals, dtype=np.float64)
        row = np.asarray(rows)
        col = np.asarray(cols)
        doc_len = np.asarray([len(d) for d in docs], dtype=np.float64)
        norm = k1 * (1 - b + b * doc_len / doc_len.mean())
        df = np.bincount(col, minlength=len(self.vocab))
        idf = np.log1p((n_docs - df + 0.5) / (df + 0.5))
        weights = idf[col] * tf * (k1 + 1) / (tf + norm[row])
        self.weights = sparse.csr_matrix((weights, (row, col)), shape=(n_docs, len(self.vocab)))

    def query_matrix(self, queries: list[list[str]]) -> sparse.csr_matrix:
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        for i, toks in enumerate(queries):
            for term, count in Counter(toks).items():
                j = self.vocab.get(term)
                if j is not None:
                    rows.append(i)
                    cols.append(j)
                    vals.append(count)
        return sparse.csr_matrix((vals, (rows, cols)), shape=(len(queries), len(self.vocab)))

    def scores(self, queries: list[list[str]]) -> np.ndarray:
        return np.asarray((self.query_matrix(queries) @ self.weights.T).todense())
