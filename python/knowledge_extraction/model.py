from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    block_size: int = 64
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = (
            z.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
            for z in self.qkv(x).split(c, dim=2)
        )
        y = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        out: torch.Tensor = self.proj(y.transpose(1, 2).reshape(b, t, c))
        return out


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.fc1 = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.fc2 = nn.Linear(4 * cfg.n_embd, cfg.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        out: torch.Tensor = x + self.fc2(nn.functional.gelu(self.fc1(self.ln2(x))))
        return out


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear | nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def residual_stream(self, idx: torch.Tensor) -> list[torch.Tensor]:
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        states = [x]
        for block in self.blocks:
            x = block(x)
            states.append(x)
        return states

    def hidden(self, idx: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.ln_f(self.residual_stream(idx)[-1])
        return out

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.head(self.hidden(idx))
        return out
