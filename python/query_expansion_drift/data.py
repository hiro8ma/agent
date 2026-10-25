from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
SCIFACT_SHA256 = "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165"


@dataclass(frozen=True)
class Doc:
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class Dataset:
    docs: list[Doc]
    queries: dict[str, str]
    train_qrels: dict[str, dict[str, int]]
    test_qrels: dict[str, dict[str, int]]


def download(path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        with urllib.request.urlopen(SCIFACT_URL, timeout=120) as res:
            tmp.write_bytes(res.read())
        tmp.rename(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SCIFACT_SHA256:
        raise ValueError(f"sha256 mismatch for {path}: {digest}")
    return path


def _read_qrels(raw: bytes) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8")), delimiter="\t")
    for row in reader:
        qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    return qrels


def load_scifact(path: Path) -> Dataset:
    with zipfile.ZipFile(download(path)) as zf:
        docs = [
            Doc(d["_id"], d["title"], d["text"])
            for d in map(json.loads, zf.read("scifact/corpus.jsonl").splitlines())
        ]
        queries = {
            q["_id"]: q["text"]
            for q in map(json.loads, zf.read("scifact/queries.jsonl").splitlines())
        }
        train_qrels = _read_qrels(zf.read("scifact/qrels/train.tsv"))
        test_qrels = _read_qrels(zf.read("scifact/qrels/test.tsv"))
    return Dataset(docs, queries, train_qrels, test_qrels)


def sample_ids(ids: list[str], n: int, seed: int) -> list[str]:
    ordered = sorted(ids, key=int)
    return random.Random(seed).sample(ordered, len(ordered))[:n]
