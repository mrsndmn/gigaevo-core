"""Small decoder-only LM with a factorized embedding and a factorized softmax.

The model is deliberately boring (pre-LN, RoPE, SwiGLU-free MLP) so that any
difference between arms comes from the vocabulary treatment and not from
architecture noise. Two things are not boring:

**Factorized softmax.** Each head in ``heads`` gets its own output projection.
With ``condition_on_stem`` the feature heads see ``h + E_stem[target]``, giving

    P(token) = P(stem | h) * P(role | h, stem) * P(suffix | h, stem)

rather than assuming the heads are conditionally independent given ``h``. The
loss is the sum over heads, so no information is obtained for free: an arm
that moves case out of the stem vocabulary still pays for predicting it.

**Character n-gram bag.** Optional. Each stem id contributes, in addition to
its own row of ``E_stem``, the mean of hashed character n-gram embeddings of
its surface string, projected up from a low rank. Two properties matter: it
ties the representations of strings that share substrings (so a typo lands
near its source), and it costs ``buckets * rank + rank * d`` parameters rather
than a full extra embedding table. The bag is computed once per forward over
the whole vocabulary, not per token, which makes it nearly free.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

LN2 = math.log(2.0)


# ---------------------------------------------------------------------------
# Transformer internals
# ---------------------------------------------------------------------------


def _rope_cache(seq: int, head_dim: int, device, dtype):
    inv = 1.0 / (
        10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    t = torch.arange(seq, device=device).float()
    freqs = torch.outer(t, inv)
    return torch.cos(freqs).to(dtype), torch.sin(freqs).to(dtype)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, D)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[None, None, : x.shape[2], :]
    sin = sin[None, None, : x.shape[2], :]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


class Block(nn.Module):
    def __init__(self, d: int, n_head: int, d_ff: int, dropout: float):
        super().__init__()
        self.n_head = n_head
        self.d_head = d // n_head
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d, bias=False)
        self.drop = nn.Dropout(dropout) if dropout else nn.Identity()

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(D, dim=2)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.drop(self.proj(a))
        h = self.ln2(x)
        x = x + self.drop(self.fc2(F.gelu(self.fc1(h))))
        return x


# ---------------------------------------------------------------------------
# LM
# ---------------------------------------------------------------------------


class FactorizedLM(nn.Module):
    def __init__(
        self,
        heads: dict[str, int],
        d_model: int = 512,
        n_layer: int = 8,
        n_head: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.0,
        tie_embeddings: bool = True,
        condition_on_stem: bool = True,
        ngram_table: torch.Tensor | None = None,  # (V, K) int64, 0 == pad
        ngram_buckets: int = 8192,
        ngram_rank: int = 64,
    ):
        super().__init__()
        self.heads = dict(heads)
        self.head_names = list(heads)
        self.d_model = d_model
        self.condition_on_stem = condition_on_stem and len(heads) > 1

        self.emb = nn.ModuleDict(
            {name: nn.Embedding(card, d_model) for name, card in heads.items()}
        )
        self.blocks = nn.ModuleList(
            [Block(d_model, n_head, d_ff, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.out = nn.ModuleDict(
            {name: nn.Linear(d_model, card, bias=False) for name, card in heads.items()}
        )
        if tie_embeddings:
            self.out["stem"].weight = self.emb["stem"].weight

        self.use_ngram = ngram_table is not None
        if self.use_ngram:
            self.register_buffer("ngram_ids", ngram_table, persistent=True)
            self.register_buffer(
                "ngram_mask", (ngram_table > 0).to(torch.float32), persistent=True
            )
            self.ngram_emb = nn.Embedding(ngram_buckets, ngram_rank, padding_idx=0)
            self.ngram_proj = nn.Linear(ngram_rank, d_model, bias=False)
            nn.init.zeros_(self.ngram_proj.weight)  # start as a no-op

        self.apply(self._init)
        self._rope: tuple[torch.Tensor, torch.Tensor] | None = None
        self._rope_len = 0

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def _ngram_bag(self) -> torch.Tensor:
        """(V, d) additive term per stem id, computed once per forward."""
        e = self.ngram_emb(self.ngram_ids)  # (V, K, r)
        m = self.ngram_mask.unsqueeze(-1)
        bag = (e * m).sum(1) / m.sum(1).clamp(min=1.0)
        return self.ngram_proj(bag)

    def forward(
        self, inputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor] | None
    ):
        stem = inputs["stem"]
        B, T = stem.shape
        x = self.emb["stem"](stem)
        if self.use_ngram:
            x = x + self._ngram_bag()[stem]
        for name in self.head_names:
            if name != "stem":
                x = x + self.emb[name](inputs[name])

        dev, dt = x.device, x.dtype
        if self._rope is None or self._rope_len < T or self._rope[0].device != dev:
            cos, sin = _rope_cache(max(T, 1024), self.blocks[0].d_head, dev, dt)
            self._rope, self._rope_len = (cos, sin), max(T, 1024)
        cos, sin = self._rope
        cos, sin = cos.to(dt), sin.to(dt)

        for blk in self.blocks:
            x = blk(x, cos, sin)
        h = self.ln_f(x)

        if targets is None:
            return {name: self.out[name](h) for name in self.head_names}

        losses: dict[str, torch.Tensor] = {}
        stem_logits = self.out["stem"](h)
        losses["stem"] = F.cross_entropy(
            stem_logits.float().view(-1, stem_logits.shape[-1]),
            targets["stem"].reshape(-1),
            reduction="sum",
        )
        if len(self.head_names) > 1:
            hf = h
            if self.condition_on_stem:
                # Teacher-forced conditioning on the stem actually being
                # predicted at this position. clamp() keeps masked targets
                # (-100, used by the corrupted-context evaluator) in range;
                # their loss is discarded by cross_entropy's ignore_index.
                hf = h + self.emb["stem"](targets["stem"].clamp(min=0))
            for name in self.head_names:
                if name == "stem":
                    continue
                lg = self.out[name](hf)
                losses[name] = F.cross_entropy(
                    lg.float().view(-1, lg.shape[-1]),
                    targets[name].reshape(-1),
                    reduction="sum",
                )
        return losses

    # -- accounting ---------------------------------------------------------

    def param_counts(self) -> dict[str, int]:
        emb = sum(p.numel() for p in self.emb.parameters())
        out = sum(
            p.numel()
            for n, mod in self.out.items()
            for p in mod.parameters()
            if not (n == "stem" and mod.weight is self.emb["stem"].weight)
        )
        ng = 0
        if self.use_ngram:
            ng = sum(p.numel() for p in self.ngram_emb.parameters()) + sum(
                p.numel() for p in self.ngram_proj.parameters()
            )
        body = sum(p.numel() for p in self.blocks.parameters()) + sum(
            p.numel() for p in self.ln_f.parameters()
        )
        total = sum(p.numel() for p in self.parameters())
        return {
            "embedding": emb,
            "output_heads": out,
            "ngram": ng,
            "body": body,
            "total": total,
        }


# ---------------------------------------------------------------------------
# n-gram table construction
# ---------------------------------------------------------------------------


def build_ngram_table(
    arm, buckets: int = 8192, k: int = 16, n_min: int = 3, n_max: int = 5
) -> torch.Tensor:
    """Map every stem id to up to ``k`` hashed character n-gram buckets.

    Bucket 0 is reserved as padding, so a hash never lands there.
    """
    import numpy as np

    V = arm.heads["stem"]
    table = np.zeros((V, k), dtype=np.int64)
    for vid in range(V):
        try:
            s = (
                arm.tok.decode([vid], skip_special_tokens=False)
                if hasattr(arm, "tok")
                else chr(vid)
            )
        except Exception:
            s = ""
        if not s:
            continue
        padded = f"<{s}>"
        grams = [
            padded[i : i + n]
            for n in range(n_min, n_max + 1)
            for i in range(len(padded) - n + 1)
        ]
        if not grams:
            continue
        ids = [(hash_str(g) % (buckets - 1)) + 1 for g in grams[:k]]
        table[vid, : len(ids)] = ids
    return torch.from_numpy(table)


def hash_str(s: str) -> int:
    """Stable FNV-1a, so tables are reproducible across processes and runs."""
    h = 0xCBF29CE484222325
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h
