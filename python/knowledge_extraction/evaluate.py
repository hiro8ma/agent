from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from data import ITEMS, Person, Vocab, parse_bio, qa_prompt, reverse_prompt
from model import GPT


@torch.no_grad()
def greedy(model: GPT, prompts: torch.Tensor, n_new: int, chunk: int = 512) -> torch.Tensor:
    outs = []
    for start in range(0, prompts.shape[0], chunk):
        seq = prompts[start : start + chunk]
        for _ in range(n_new):
            nxt = model(seq)[:, -1].argmax(dim=-1, keepdim=True)
            seq = torch.cat([seq, nxt], dim=1)
        outs.append(seq[:, prompts.shape[1] :])
    return torch.cat(outs)


def _prompts(vocab: Vocab, prompts: Sequence[Sequence[str]]) -> torch.Tensor:
    return torch.tensor([[vocab.bos, *vocab.encode(p)] for p in prompts])


def answer_correct(
    model: GPT,
    vocab: Vocab,
    prompts: Sequence[Sequence[str]],
    answers: Sequence[Sequence[str]],
) -> np.ndarray:
    width = max(len(a) for a in answers)
    out = greedy(model, _prompts(vocab, prompts), width).tolist()
    return np.array(
        [vocab.decode(o[: len(a)]) == list(a) for o, a in zip(out, answers, strict=True)]
    )


def qa_correct(model: GPT, vocab: Vocab, people: Sequence[Person], item: str) -> np.ndarray:
    return answer_correct(
        model, vocab, [qa_prompt(p, item) for p in people], [p.value(item) for p in people]
    )


def reverse_correct(model: GPT, vocab: Vocab, people: Sequence[Person]) -> np.ndarray:
    return answer_correct(
        model, vocab, [reverse_prompt(p) for p in people], [p.name for p in people]
    )


def bio_correct(
    model: GPT, vocab: Vocab, people: Sequence[Person], max_new: int
) -> dict[str, np.ndarray]:
    out = greedy(model, _prompts(vocab, [p.name for p in people]), max_new).tolist()
    correct: dict[str, list[bool]] = {item: [] for item in ITEMS}
    for person, ids in zip(people, out, strict=True):
        facts: dict[str, tuple[str, ...]] = {}
        for item, value in parse_bio([*person.name, *vocab.decode(ids)]):
            facts.setdefault(item, value)
        for item in ITEMS:
            correct[item].append(facts.get(item) == person.value(item))
    return {item: np.array(v) for item, v in correct.items()}
