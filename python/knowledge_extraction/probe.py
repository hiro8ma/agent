from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

from data import Person, Vocab, n_classes
from model import GPT

WEIGHT_DECAYS = (1e-4, 1e-3, 1e-2, 1e-1)


def name_end_index(ids: Sequence[int], name_ids: Sequence[int]) -> int:
    n = len(name_ids)
    for i in range(len(ids) - n + 1):
        if list(ids[i : i + n]) == list(name_ids):
            return i + n - 1
    raise ValueError("name not found")


@torch.no_grad()
def name_features(model: GPT, vocab: Vocab, people: Sequence[Person], layer: int) -> torch.Tensor:
    rows = [[vocab.bos, *vocab.encode(p.name)] for p in people]
    pos = torch.tensor(
        [name_end_index(r, vocab.encode(p.name)) for r, p in zip(rows, people, strict=True)]
    )
    states = model.residual_stream(torch.tensor(rows))[layer]
    return states[torch.arange(len(rows)), pos]


def _fit(
    x: torch.Tensor,
    y: torch.Tensor,
    n_class: int,
    weight_decay: float,
    steps: int,
    seed: int,
    hidden: int,
) -> nn.Module:
    torch.manual_seed(seed)
    clf: nn.Module = (
        nn.Linear(x.shape[1], n_class)
        if hidden == 0
        else nn.Sequential(nn.Linear(x.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, n_class))
    )
    weights = [p for p in clf.parameters() if p.dim() == 2]
    opt = torch.optim.Adam(clf.parameters(), lr=0.05 if hidden == 0 else 0.01)
    for _ in range(steps):
        l2 = sum(w.square().sum() for w in weights)
        loss = nn.functional.cross_entropy(clf(x), y) + weight_decay * l2
        opt.zero_grad(set_to_none=True)
        torch.autograd.backward(loss)
        opt.step()
    return clf


def probe_predict(
    x_train: torch.Tensor,
    y_train: np.ndarray,
    x_test: torch.Tensor,
    n_class: int,
    seed: int,
    hidden: int = 0,
    steps: int = 300,
) -> np.ndarray:
    mean, std = x_train.mean(dim=0), x_train.std(dim=0) + 1e-6
    xtr, xte = (x_train - mean) / std, (x_test - mean) / std
    ytr = torch.from_numpy(y_train)
    order = np.random.default_rng(seed).permutation(len(y_train))
    fit_idx, val_idx = order[: len(order) * 4 // 5], order[len(order) * 4 // 5 :]
    best, best_acc = WEIGHT_DECAYS[0], -1.0
    for wd in WEIGHT_DECAYS:
        clf = _fit(xtr[fit_idx], ytr[fit_idx], n_class, wd, steps, seed, hidden)
        with torch.no_grad():
            acc = float((clf(xtr[val_idx]).argmax(dim=1) == ytr[val_idx]).float().mean())
        if acc > best_acc:
            best, best_acc = wd, acc
    clf = _fit(xtr, ytr, n_class, best, steps, seed, hidden)
    with torch.no_grad():
        pred: np.ndarray = clf(xte).argmax(dim=1).numpy()
    return pred


def probe_correct(
    model: GPT,
    vocab: Vocab,
    train_people: Sequence[Person],
    test_people: Sequence[Person],
    item: str,
    layer: int,
    seed: int,
    hidden: int = 0,
) -> np.ndarray:
    x_train = name_features(model, vocab, train_people, layer)
    x_test = name_features(model, vocab, test_people, layer)
    correct = np.ones(len(test_people), dtype=bool)
    for k, n_class in enumerate(n_classes(item)):
        y_train = np.array([p.labels(item)[k] for p in train_people])
        y_test = np.array([p.labels(item)[k] for p in test_people])
        correct &= probe_predict(x_train, y_train, x_test, n_class, seed, hidden) == y_test
    return correct


def majority_correct(
    train_people: Sequence[Person], test_people: Sequence[Person], item: str
) -> np.ndarray:
    correct = np.ones(len(test_people), dtype=bool)
    for k in range(len(n_classes(item))):
        majority = int(np.argmax(np.bincount([p.labels(item)[k] for p in train_people])))
        correct &= np.array([p.labels(item)[k] for p in test_people]) == majority
    return correct
