from __future__ import annotations

import torch


def candidate_log_q(positive_freq: torch.Tensor, n_in_batch: int, n_uniform: int) -> torch.Tensor:
    n_items = positive_freq.numel()
    q = (n_in_batch * positive_freq + n_uniform / n_items) / (n_in_batch + n_uniform)
    return torch.log(q)


def build_logits(
    scores: torch.Tensor,
    positive_ids: torch.Tensor,
    candidate_ids: torch.Tensor,
    log_q: torch.Tensor | None = None,
    remove_false_negatives: bool = False,
) -> torch.Tensor:
    batch = positive_ids.numel()
    if not torch.equal(candidate_ids[:batch], positive_ids):
        raise ValueError("the first columns must be the in-batch positives")
    logits = scores
    if log_q is not None:
        logits = logits - log_q[candidate_ids].unsqueeze(0)
    if remove_false_negatives:
        hit = positive_ids.unsqueeze(1) == candidate_ids.unsqueeze(0)
        hit[:, :batch].fill_diagonal_(False)
        logits = logits.masked_fill(hit, float("-inf"))
    return logits


def contrastive_loss(
    scores: torch.Tensor,
    positive_ids: torch.Tensor,
    candidate_ids: torch.Tensor,
    log_q: torch.Tensor | None = None,
    remove_false_negatives: bool = False,
) -> torch.Tensor:
    logits = build_logits(scores, positive_ids, candidate_ids, log_q, remove_false_negatives)
    target = torch.arange(positive_ids.numel(), device=scores.device)
    return torch.nn.functional.cross_entropy(logits, target)


def info_nce_bound(pos_logit: torch.Tensor, neg_logits: torch.Tensor) -> torch.Tensor:
    n = neg_logits.shape[1] + 1
    all_logits = torch.cat([pos_logit.unsqueeze(1), neg_logits], dim=1)
    loss = (torch.logsumexp(all_logits, dim=1) - pos_logit).mean()
    return torch.log(torch.tensor(float(n))) - loss
