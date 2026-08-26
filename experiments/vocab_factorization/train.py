"""Train one experiment from ``configs.EXPERIMENTS`` and report bits-per-byte.

The comparability rules this file enforces:

* **Compute is budgeted in bytes of text, not tokens.** ``train_bytes`` is
  fixed across arms, so a coarser tokenizer gets fewer optimizer steps over
  the same text rather than more data.
* **BPB counts every head.** Total NLL is summed over the stem head and all
  feature heads before dividing by bytes, so moving case or morphology into a
  side head never looks free.
* **The denominator is the bytes actually scored.** Eval windows overlap by
  one token, so every token except the corpus's first is predicted exactly
  once, and the first token's bytes are excluded.

Known confound, reported rather than hidden: sequence length is fixed at 1024
*tokens* for every arm, so a finer tokenizer sees fewer bytes of context. The
``context_bytes`` field in the results records how much.

Run::

    python -m experiments.vocab_factorization.train --exp fact16k
    python -m experiments.vocab_factorization.train --all
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from experiments.vocab_factorization import perturb
from experiments.vocab_factorization.build_arms import get_arm
from experiments.vocab_factorization.configs import (
    EXPERIMENTS,
    EXPERIMENTS_BY_NAME,
    Experiment,
)
from experiments.vocab_factorization.model import (
    LN2,
    FactorizedLM,
    build_ngram_table,
)
from experiments.vocab_factorization.prepare import build_text, encode_text, get_tokens

EVAL_SUITES = ("clean", "typo1", "typo5", "lower", "upper", "title", "randcase")


def _heads_arrays(enc, heads: dict[str, int]) -> dict[str, np.ndarray]:
    src = {"stem": enc.stem, "role": enc.role, "suffix": enc.suffix}
    return {name: src[name].astype(np.int64) for name in heads}


class WindowSampler:
    """Random fixed-length windows from the token stream."""

    def __init__(self, arrays: dict[str, np.ndarray], seq_len: int, seed: int, device):
        # The whole token stream lives on the GPU (a few hundred MB), so batch
        # assembly is a device-side gather instead of a host->device copy in
        # the training loop's critical path.
        self.arrays = {k: torch.from_numpy(v).to(device) for k, v in arrays.items()}
        self.n = next(iter(arrays.values())).shape[0]
        self.seq_len = seq_len
        self.device = device
        self.g = torch.Generator(device=device).manual_seed(seed)
        self.ar = torch.arange(seq_len + 1, device=device)

    def batch(self, batch_size: int):
        idx = torch.randint(
            0,
            self.n - self.seq_len - 1,
            (batch_size,),
            generator=self.g,
            device=self.device,
        )
        offs = idx[:, None] + self.ar[None, :]
        inputs, targets = {}, {}
        for name, arr in self.arrays.items():
            w = arr[offs]
            inputs[name] = w[:, :-1]
            targets[name] = w[:, 1:]
        return inputs, targets


@torch.no_grad()
def eval_bpb(
    model: FactorizedLM,
    arrays: dict[str, np.ndarray],
    nbytes: int,
    seq_len: int,
    device,
    batch_size: int = 8,
) -> dict[str, float]:
    """Bits per byte over the whole array, windows overlapping by one token."""
    model.eval()
    n = next(iter(arrays.values())).shape[0]
    tensors = {k: torch.from_numpy(v) for k, v in arrays.items()}
    stride = seq_len  # window covers seq_len+1 tokens, scores seq_len of them
    starts = list(range(0, n - seq_len - 1, stride))
    per_head = {name: 0.0 for name in model.head_names}
    scored = 0

    for i in range(0, len(starts), batch_size):
        chunk = starts[i : i + batch_size]
        offs = torch.tensor(chunk)[:, None] + torch.arange(seq_len + 1)[None, :]
        inputs, targets = {}, {}
        for name, arr in tensors.items():
            w = arr[offs]
            inputs[name] = w[:, :-1].to(device)
            targets[name] = w[:, 1:].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            losses = model(inputs, targets)
        for name, v in losses.items():
            per_head[name] += float(v)
        scored += len(chunk) * seq_len

    total_nll = sum(per_head.values())
    # Bytes attributable to the scored tokens.
    bytes_scored = nbytes * scored / n
    out = {
        "bpb": total_nll / LN2 / bytes_scored,
        "tokens_scored": scored,
        "bytes_scored": bytes_scored,
    }
    for name, v in per_head.items():
        out[f"bpb.{name}"] = v / LN2 / bytes_scored
    model.train()
    return out


def run(exp: Experiment, args) -> dict:
    device = torch.device("cuda")
    torch.manual_seed(exp.train.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    arm = get_arm(exp.arm, args.arms_dir, args.corpus)
    heads = arm.heads

    budget = args.train_mb * 1024 * 1024 if args.train_mb else exp.train.train_bytes
    enc, train_bytes = get_tokens(
        exp.arm,
        "train",
        budget,
        args.corpus,
        args.arms_dir,
        args.tokens_dir,
        workers=args.workers,
    )
    train_arrays = _heads_arrays(enc, heads)
    n_train_tokens = len(enc)

    ngram_table = None
    if exp.model.ngram_bag:
        ngram_table = build_ngram_table(
            arm,
            buckets=exp.model.ngram_buckets,
            n_min=exp.model.ngram_min,
            n_max=exp.model.ngram_max,
        )

    model = FactorizedLM(
        heads=heads,
        d_model=exp.model.d_model,
        n_layer=exp.model.n_layer,
        n_head=exp.model.n_head,
        d_ff=exp.model.d_ff,
        dropout=exp.model.dropout,
        tie_embeddings=exp.model.tie_embeddings,
        condition_on_stem=exp.model.condition_on_stem,
        ngram_table=ngram_table,
        ngram_buckets=exp.model.ngram_buckets,
        ngram_rank=exp.model.ngram_rank,
    ).to(device)
    counts = model.param_counts()

    micro = max(1, args.micro_batch)
    seq = exp.train.seq_len
    accum = max(1, exp.train.batch_tokens // (micro * seq))
    steps = args.steps or int(
        exp.train.epochs * n_train_tokens / exp.train.batch_tokens
    )
    warmup = max(1, int(steps * exp.train.warmup_frac))

    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": exp.train.weight_decay},
            {"params": nodecay, "weight_decay": 0.0},
        ],
        lr=exp.train.lr,
        betas=(0.9, 0.95),
    )

    def lr_at(s: int) -> float:
        if s < warmup:
            return exp.train.lr * (s + 1) / warmup
        p = (s - warmup) / max(1, steps - warmup)
        f = exp.train.min_lr_frac
        return exp.train.lr * (f + (1 - f) * 0.5 * (1 + math.cos(math.pi * p)))

    sampler = WindowSampler(train_arrays, seq, exp.train.seed, device)

    print(
        f"[{exp.name}] arm={exp.arm} heads={heads} "
        f"tokens={n_train_tokens / 1e6:.1f}M bytes={train_bytes / 1e6:.1f}MB "
        f"steps={steps} accum={accum} params={counts}",
        flush=True,
    )

    model.train()
    t0 = time.time()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        tot = torch.zeros((), device=device)
        for _ in range(accum):
            inputs, targets = sampler.batch(micro)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                losses = model(inputs, targets)
            loss = sum(losses.values()) / (micro * seq * accum)
            loss.backward()
            # Accumulate on device: float(loss) here would sync the GPU once
            # per micro-batch and dominate step time.
            tot += loss.detach()
        torch.nn.utils.clip_grad_norm_(model.parameters(), exp.train.grad_clip)
        opt.step()
        if step % exp.train.log_every == 0 or step == steps - 1:
            el = time.time() - t0
            print(
                f"[{exp.name}] step {step:5d}/{steps} loss/token={float(tot):.4f} "
                f"lr={lr_at(step):.2e} {el:6.1f}s",
                flush=True,
            )
    train_seconds = time.time() - t0

    # ---- evaluation ------------------------------------------------------
    valid_text = build_text(args.corpus, "valid", None)
    if args.eval_mb:
        cut = int(args.eval_mb * 1024 * 1024)
        valid_text = valid_text[:cut]
        valid_text = valid_text[: valid_text.rfind("\n") + 1]

    results: dict[str, object] = {
        "experiment": exp.name,
        "arm": exp.arm,
        "heads": heads,
        "params": counts,
        "steps": steps,
        "train_tokens": n_train_tokens,
        "train_bytes": train_bytes,
        "train_seconds": round(train_seconds, 1),
        "tokens_per_byte_train": n_train_tokens / train_bytes,
        "context_bytes": seq / (n_train_tokens / train_bytes),
        "note": exp.note,
    }

    for suite in EVAL_SUITES:
        text = perturb.apply_suite(suite, valid_text, seed=0)
        e = encode_text(exp.arm, args.arms_dir, text, workers=args.workers)
        nb = len(text.encode("utf-8"))
        m = eval_bpb(model, _heads_arrays(e, heads), nb, seq, device, args.eval_batch)
        results[f"eval.{suite}"] = m
        print(
            f"[{exp.name}] {suite:9s} bpb={m['bpb']:.4f} "
            + " ".join(
                f"{k.split('.')[1]}={v:.4f}"
                for k, v in m.items()
                if k.startswith("bpb.")
            ),
            flush=True,
        )

    clean = results["eval.clean"]["bpb"]
    for suite in EVAL_SUITES:
        results[f"delta.{suite}"] = results[f"eval.{suite}"]["bpb"] - clean

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{exp.name}.json").write_text(json.dumps(results, indent=2))
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    ap.add_argument("--arms-dir", type=Path, default=Path("data/arms"))
    ap.add_argument("--tokens-dir", type=Path, default=Path("data/tokens"))
    ap.add_argument("--out", type=Path, default=Path("results/runs"))
    ap.add_argument(
        "--steps", type=int, default=0, help="override step count (smoke tests)"
    )
    ap.add_argument(
        "--train-mb", type=int, default=0, help="override train_bytes (smoke tests)"
    )
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--eval-batch", type=int, default=8)
    ap.add_argument("--eval-mb", type=float, default=0.0, help="0 = whole valid split")
    ap.add_argument("--workers", type=int, default=48)
    args = ap.parse_args()

    if args.all:
        todo = EXPERIMENTS
    elif args.exp:
        todo = [EXPERIMENTS_BY_NAME[n] for n in args.exp]
    else:
        raise SystemExit("pass --exp NAME [NAME ...] or --all")

    for exp in todo:
        run(exp, args)


if __name__ == "__main__":
    main()
