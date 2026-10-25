from __future__ import annotations

import gzip
import json
import random
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DATA_URL = (
    "https://raw.githubusercontent.com/nelson-liu/lost-in-the-middle/main/"
    "qa_data/20_total_documents/nq-open-20_total_documents_gold_at_0.jsonl.gz"
)
DATA_FILE = "nq-open-20_total_documents_gold_at_0.jsonl.gz"


@dataclass(frozen=True)
class Document:
    title: str
    text: str


@dataclass(frozen=True)
class Example:
    question: str
    answers: tuple[str, ...]
    gold: Document
    distractors: tuple[Document, ...]


def download(data_dir: Path) -> Path:
    path = data_dir / DATA_FILE
    if path.exists():
        return path
    data_dir.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")
    urllib.request.urlretrieve(DATA_URL, partial)
    partial.rename(path)
    return path


def parse_example(row: Mapping[str, Any]) -> Example:
    ctxs = row["ctxs"]
    docs = [Document(title=str(c["title"]), text=str(c["text"])) for c in ctxs]
    golds = [i for i, c in enumerate(ctxs) if c["isgold"]]
    if golds != [0]:
        raise ValueError(f"gold_at_0 のファイルなのに正解文書の位置が {golds}")
    return Example(
        question=str(row["question"]),
        answers=tuple(str(a) for a in row["answers"]),
        gold=docs[0],
        distractors=tuple(docs[1:]),
    )


def load_examples(path: Path) -> list[Example]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [parse_example(json.loads(line)) for line in f]


def sample_examples(examples: list[Example], n: int, seed: int) -> list[Example]:
    return random.Random(seed).sample(examples, n)
