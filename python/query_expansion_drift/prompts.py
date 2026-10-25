from __future__ import annotations

import random
from dataclasses import dataclass

from data import Dataset

FORMS = ("query2doc", "hyde", "properties")
MAX_TOKENS = {"query2doc": 128, "hyde": 128, "properties": 64}


@dataclass(frozen=True)
class Example:
    claim: str
    passage: str


def truncate_words(text: str, n: int) -> str:
    words = text.split()
    return " ".join(words[:n])


def pick_examples(ds: Dataset, n: int, seed: int, max_words: int = 60) -> list[Example]:
    test_docs = {d for rels in ds.test_qrels.values() for d in rels}
    by_id = {d.doc_id: d for d in ds.docs}
    candidates = []
    for qid in sorted(ds.train_qrels, key=int):
        rels = sorted(ds.train_qrels[qid], key=int)
        if not rels or any(r in test_docs for r in rels):
            continue
        candidates.append((qid, rels[0]))
    chosen = random.Random(seed).sample(candidates, n)
    return [
        Example(ds.queries[qid], truncate_words(by_id[did].text, max_words)) for qid, did in chosen
    ]


def query2doc_prompt(claim: str, examples: list[Example]) -> str:
    shots = "\n\n".join(f"Claim: {e.claim}\nPassage: {e.passage}" for e in examples)
    return (
        "Write a passage from a scientific abstract that addresses the given claim. "
        "Reply with the passage only.\n\n"
        f"{shots}\n\nClaim: {claim}\nPassage:"
    )


def hyde_prompt(claim: str) -> str:
    return (
        "Please write a scientific paper passage to support/refute the claim. "
        "Reply with the passage only.\n"
        f"Claim: {claim}\nPassage:"
    )


def properties_prompt(claim: str) -> str:
    return (
        "List concrete keywords that a scientific abstract relevant to the claim would contain: "
        "the specific genes, proteins, drugs, diseases, cell types, organisms and outcomes involved, "
        "plus closely related terms and synonyms. Reply with a comma-separated list only, no sentences.\n"
        f"Claim: {claim}\nKeywords:"
    )


def build_prompt(form: str, claim: str, examples: list[Example]) -> str:
    if form == "query2doc":
        return query2doc_prompt(claim, examples)
    if form == "hyde":
        return hyde_prompt(claim)
    if form == "properties":
        return properties_prompt(claim)
    raise ValueError(form)


def clean(text: str) -> str:
    text = text.strip()
    for prefix in ("Passage:", "Keywords:"):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    return text
