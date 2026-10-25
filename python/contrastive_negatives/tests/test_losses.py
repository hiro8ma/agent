from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from data import count_overlapping_rows
from losses import build_logits, candidate_log_q, contrastive_loss, info_nce_bound


def _batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positive = torch.tensor([3, 5, 3, 7])
    uniform = torch.tensor([5, 9, 3])
    candidates = torch.cat([positive, uniform])
    scores = torch.arange(4 * 7, dtype=torch.float32).reshape(4, 7) / 10
    return scores, positive, candidates


def test_false_negative_removal_masks_repeats_and_keeps_diagonal() -> None:
    scores, positive, candidates = _batch()

    logits = build_logits(scores, positive, candidates, remove_false_negatives=True)

    masked = {(r, c) for r, c in torch.isinf(logits).nonzero().tolist()}
    assert masked == {(0, 2), (0, 6), (1, 4), (2, 0), (2, 6)}
    assert torch.isfinite(torch.diagonal(logits[:, :4])).all()
    assert torch.isfinite(
        contrastive_loss(scores, positive, candidates, remove_false_negatives=True)
    )


def test_without_removal_nothing_is_masked() -> None:
    scores, positive, candidates = _batch()

    logits = build_logits(scores, positive, candidates)

    assert torch.isfinite(logits).all()


def test_removal_lowers_loss_when_the_positive_repeats() -> None:
    positive = torch.tensor([3, 3])
    scores = torch.zeros(2, 2)

    kept = contrastive_loss(scores, positive, positive)
    removed = contrastive_loss(scores, positive, positive, remove_false_negatives=True)

    assert kept.item() == pytest.approx(math.log(2))
    assert removed.item() == pytest.approx(0.0)


def test_log_q_penalises_frequent_candidates() -> None:
    positive = torch.tensor([0, 1])
    candidates = torch.tensor([0, 1])
    log_q = torch.log(torch.tensor([0.9, 0.1]))
    scores = torch.zeros(2, 2)

    logits = build_logits(scores, positive, candidates, log_q=log_q)

    assert (logits[:, 0] < logits[:, 1]).all()
    assert (logits[:, 1] - logits[:, 0]).tolist() == pytest.approx([math.log(9.0)] * 2)


def test_candidate_log_q_mixes_in_batch_and_uniform_by_count() -> None:
    freq = torch.tensor([0.75, 0.25, 0.0, 0.0])

    in_batch_only = candidate_log_q(freq, n_in_batch=8, n_uniform=0)
    mixed = candidate_log_q(freq, n_in_batch=8, n_uniform=8)

    assert in_batch_only[:2].exp().tolist() == pytest.approx([0.75, 0.25])
    assert mixed.exp().tolist() == pytest.approx([0.5, 0.25, 0.125, 0.125])
    assert mixed.exp().sum().item() == pytest.approx(1.0)


def test_info_nce_bound_cannot_exceed_log_n() -> None:
    pos = torch.full((5,), 50.0)
    neg = torch.full((5, 7), -50.0)

    bound = info_nce_bound(pos, neg)

    assert bound.item() == pytest.approx(math.log(8), abs=1e-6)


def test_count_overlapping_rows_counts_every_row_of_a_repeated_item() -> None:
    assert count_overlapping_rows(np.array([1, 2, 1, 3, 4, 4, 5])) == 4
