from __future__ import annotations

import torch
from torch import nn


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))


class TwoTower(nn.Module):
    features: torch.Tensor

    def __init__(self, features: torch.Tensor, hidden: int, out_dim: int) -> None:
        super().__init__()
        n_items, in_dim = features.shape
        self.register_buffer("features", features)
        self.query_mlp = _mlp(in_dim, hidden, out_dim)
        self.item_mlp = _mlp(in_dim, hidden, out_dim)
        # popularity is not in the features, so items need an ID embedding to express it
        self.item_id = nn.Embedding(n_items, out_dim)
        nn.init.zeros_(self.item_id.weight)

    def query(self, ids: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.query_mlp(self.features[ids]), dim=-1)

    def item(self, ids: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(
            self.item_mlp(self.features[ids]) + self.item_id(ids), dim=-1
        )
