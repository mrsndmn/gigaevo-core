"""Bits-per-byte of a CLEAN continuation given a CORRUPTED context.

Why this metric exists. The study's original robustness number was BPB measured
on corrupted text, and it could not detect an input-side mechanism even in
principle: most of that cost is the model being charged for *emitting* the
corruption, which no change to the input representation can reduce. The
character n-gram bag scored 0.679 vs 0.676 against its own control -- not
because it did nothing, but because the instrument could not see it.

Here the corruption is confined to the context. Each block is

    [ corrupted prefix ]  [ clean suffix ]
      context, unscored     scored

so nothing is ever charged for reproducing noise. The only thing measured is
whether typos upstream damaged the model's ability to predict clean text
downstream -- which is what "robust to typos" actually means.

Comparability notes:

* Prefix and suffix are encoded separately and concatenated, forcing a token
  boundary at the split point. That boundary is identical for every arm, and
  it is the only way to know which tokens are clean when corruption changes
  the segmentation.
* The denominator is the bytes of the clean suffix only.
* Masked positions carry target ``-100``, which ``F.cross_entropy`` ignores, so
  the same loss path serves both evaluators.
"""

from __future__ import annotations

import numpy as np
import torch

from experiments.vocab_factorization import perturb
from experiments.vocab_factorization.model import LN2

IGNORE = -100
# Not a perturbation: replaces the context with an unrelated one, to calibrate
# how much *any* context damage can possibly cost on this model.
MISMATCH = "mismatch"


def _blocks(text: str, ctx_bytes: int, score_bytes: int) -> list[tuple[str, str]]:
    """Cut text into (context, scored) pairs on newline boundaries."""
    out: list[tuple[str, str]] = []
    i, n = 0, len(text)
    while i < n:
        j = text.find("\n", i + ctx_bytes)
        if j == -1:
            break
        k = text.find("\n", j + score_bytes)
        if k == -1:
            k = n
        ctx, scored = text[i:j], text[j:k]
        if ctx and scored:
            out.append((ctx, scored))
        i = k
    return out


def build_batches(
    arm,
    text: str,
    suite: str,
    seq_len: int,
    ctx_bytes: int = 2000,
    score_bytes: int = 2000,
    seed: int = 0,
):
    """Return (list of (inputs, targets) numpy dicts, scored byte count)."""
    heads = list(arm.heads)
    items = []
    total_bytes = 0
    blocks = _blocks(text, ctx_bytes, score_bytes)
    for bi, (ctx, scored) in enumerate(blocks):
        if suite == MISMATCH:
            # Sensitivity control: swap in an unrelated document's context.
            # If this costs no more than a typo'd context, the model is not
            # using context and the whole metric is measuring nothing.
            ctx = blocks[(bi + len(blocks) // 2) % len(blocks)][0]
            c_enc = arm.encode(ctx)
        else:
            c_enc = arm.encode(perturb.apply_suite(suite, ctx, seed))
        s_enc = arm.encode(scored)
        n_c, n_s = len(c_enc), len(s_enc)
        if n_s < 2:
            continue
        # Keep the scored part whole; drop context from the left if needed.
        if n_c + n_s > seq_len + 1:
            drop = n_c + n_s - (seq_len + 1)
            if drop >= n_c:
                continue
            n_c -= drop
        cols = {}
        for name, src in (("stem", "stem"), ("role", "role"), ("suffix", "suffix")):
            cols[name] = np.concatenate(
                [getattr(c_enc, src)[len(c_enc) - n_c :], getattr(s_enc, src)]
            ).astype(np.int64)
        n = n_c + n_s
        inputs = {h: cols[h][:-1].copy() for h in heads}
        targets = {h: cols[h][1:].copy() for h in heads}
        # Position i predicts token i+1; mask everything that lands in context.
        mask = np.arange(n - 1) + 1 < n_c
        for h in heads:
            targets[h][mask] = IGNORE
        items.append((inputs, targets, n - 1))
        total_bytes += len(scored.encode("utf-8"))
    return items, total_bytes


@torch.no_grad()
def eval_context_bpb(
    model,
    arm,
    text: str,
    suite: str,
    seq_len: int,
    device,
    batch_size: int = 8,
    ctx_bytes: int = 2000,
    score_bytes: int = 2000,
) -> dict:
    items, nbytes = build_batches(
        arm, text, suite, seq_len, ctx_bytes=ctx_bytes, score_bytes=score_bytes
    )
    if not items or nbytes == 0:
        return {"bpb": float("nan"), "blocks": 0, "bytes_scored": 0}

    model.eval()
    heads = list(arm.heads)
    per_head = dict.fromkeys(heads, 0.0)
    for i in range(0, len(items), batch_size):
        chunk = items[i : i + batch_size]
        width = max(c[2] for c in chunk)
        inp, tgt = {}, {}
        for h in heads:
            inp[h] = torch.zeros(len(chunk), width, dtype=torch.long)
            tgt[h] = torch.full((len(chunk), width), IGNORE, dtype=torch.long)
        for b, (bi, bt, ln) in enumerate(chunk):
            for h in heads:
                inp[h][b, :ln] = torch.from_numpy(bi[h])
                tgt[h][b, :ln] = torch.from_numpy(bt[h])
        inp = {h: v.to(device) for h, v in inp.items()}
        tgt = {h: v.to(device) for h, v in tgt.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            losses = model(inp, tgt)
        for h, v in losses.items():
            per_head[h] += float(v)
    model.train()

    total = sum(per_head.values())
    out = {
        "bpb": total / LN2 / nbytes,
        "blocks": len(items),
        "bytes_scored": nbytes,
    }
    for h, v in per_head.items():
        out[f"bpb.{h}"] = v / LN2 / nbytes
    return out
