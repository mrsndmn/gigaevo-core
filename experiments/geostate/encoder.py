"""The invariant-feature encoder and a minimal LoRA, in torch.

`InvariantEncoder` maps the exact invariant features of one object (its own
vector plus one vector per existing object, see `features.py`) to a vector in
the language model's embedding space: a pair MLP, attention pooling over the
existing objects, a residual MLP stack and a projector. Nothing in it is
equivariant or learned about poses; invariance comes from the features.

`LoRALinear` is a hand-rolled low-rank adapter so that the probe adds no
dependency to the shared environment; it matches peft's defaults (A
Kaiming-uniform, B zero, scaling alpha/r) and is applied identically to both
V1 arms (ADR 0006).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .features import PAIR_DIM, SELF_DIM


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.dropout = nn.Dropout(dropout)
        self.scaling = alpha / r
        self.lora_A.to(base.weight.device)
        self.lora_B.to(base.weight.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = self.lora_B(self.lora_A(self.dropout(x.to(self.lora_A.weight.dtype))))
        return out + delta.to(out.dtype) * self.scaling


def apply_lora(
    model: nn.Module, targets: tuple[str, ...], r: int, alpha: int, dropout: float
) -> list[nn.Parameter]:
    """Wrap every nn.Linear whose leaf name is in `targets`; returns the new params."""
    params: list[nn.Parameter] = []
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child_name in targets:
                wrapped = LoRALinear(child, r, alpha, dropout)
                setattr(module, child_name, wrapped)
                params.extend([wrapped.lora_A.weight, wrapped.lora_B.weight])
    if not params:
        raise ValueError(f"no linear layers named {targets} in the model")
    return params


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if ".lora_A." in name or ".lora_B." in name
    }


def _mlp(d_in: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d_out), nn.GELU(), nn.Linear(d_out, d_out))


class InvariantEncoder(nn.Module):
    def __init__(self, d_enc: int, d_model: int, n_layers: int = 2):
        super().__init__()
        self.self_mlp = _mlp(SELF_DIM, d_enc)
        self.pair_mlp = _mlp(PAIR_DIM, d_enc)
        self.pair_score = nn.Linear(d_enc, 1)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(nn.LayerNorm(d_enc), _mlp(d_enc, d_enc))
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_enc)
        self.proj = nn.Linear(d_enc, d_model)
        # Start small: the projector's output enters the LM next to token
        # embeddings whose entries are O(1e-2); a large random vector would
        # swamp the prompt early in training.
        nn.init.normal_(self.proj.weight, std=0.02 / math.sqrt(d_enc))
        nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        self_feats: torch.Tensor,  # [N, SELF_DIM]
        pair_feats: torch.Tensor,  # [N, M, PAIR_DIM]
        pair_mask: torch.Tensor,  # [N, M] bool, True where a pair exists
    ) -> torch.Tensor:  # [N, d_model]
        h_pairs = self.pair_mlp(pair_feats)
        scores = self.pair_score(h_pairs).squeeze(-1)
        scores = scores.masked_fill(~pair_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        pooled = (weights * h_pairs).sum(dim=1)
        h = self.self_mlp(self_feats) + pooled
        for block in self.blocks:
            h = h + block(h)
        return self.proj(self.norm(h))


def pad_features(
    items: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack (self [S], pairs [M_i, P]) items into padded batch tensors."""
    if not items:
        return (
            torch.zeros(0, SELF_DIM),
            torch.zeros(0, 1, PAIR_DIM),
            torch.zeros(0, 1, dtype=torch.bool),
        )
    m_max = max(p.shape[0] for _, p in items)
    self_feats = torch.stack([s for s, _ in items])
    pairs = torch.zeros(len(items), m_max, PAIR_DIM)
    mask = torch.zeros(len(items), m_max, dtype=torch.bool)
    for i, (_, p) in enumerate(items):
        pairs[i, : p.shape[0]] = p
        mask[i, : p.shape[0]] = True
    return self_feats, pairs, mask
