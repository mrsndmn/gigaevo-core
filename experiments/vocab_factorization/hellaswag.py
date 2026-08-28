"""HellaSwag loglikelihood multiple choice for transplanted models.

Study 2's headline is that a transplant *beats* the model it replaced: bpe8k-t
0.7031 clean BPB against orig-frozen's 0.7125, with 295M fewer parameters. That
number is measured on the same corpus the new embeddings were trained on, and
the plan concedes in as many words that "good BPB does not guarantee preserved
instruction-following". This module is the cheapest available thing that can
falsify it: a task the transplant was never trained for, scored the same way
for every arm.

**Gate (pre-registered).** ``bpe8k-t`` within a few points of ``orig-frozen``
on HellaSwag → study 2's headline stands. Otherwise it gets rewritten from
"beats the model it replaced" to "matches it on corpus BPB".

Scoring is the standard harness protocol: for each of the four endings, sum the
log-probability of the continuation tokens given the context, and take the
argmax. Two numbers come out, because the choice between them is not neutral --
``acc`` uses the raw sum and therefore favours short endings, while ``acc_norm``
divides by the continuation's **byte** length. Bytes rather than tokens is the
load-bearing detail: the endings are the same strings for every arm, so a byte
denominator is identical across arms, whereas a token denominator would hand a
coarser tokenizer a different normalizer and quietly make the comparison about
tokenization rather than about the embedding.

**The active-vocabulary mask is deliberately lifted here.** The mask exists so
that per-arm BPB on *this* ru+en corpus is comparable -- it removes ids the
corpus never exercises from a shared normalizer. HellaSwag is English from a
different distribution, so some of its tokens fall outside every arm's active
set; scoring them under the mask would make them impossible rather than
unlikely, and the benchmark would measure the mask instead of the model. So the
mask comes off, and the thing it would have hidden is reported instead: the
fraction of scored tokens outside each arm's active set is a first-class field
in the result. An arm that cannot represent the benchmark's text is not a
scoring inconvenience, it is a finding about the transplant.

Run::

    python -m experiments.vocab_factorization.hellaswag --native
    python -m experiments.vocab_factorization.hellaswag --ckpt <artifacts>/ckpt/bpe8k-t.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time

import numpy as np
import torch

from experiments.vocab_factorization import active_vocab, transplant
from experiments.vocab_factorization.build_arms import get_arm
from experiments.vocab_factorization.configs import (
    HOST_ARM,
    HOST_MODEL,
    TransplantTrainSpec,
)
from experiments.vocab_factorization.train_transplant import _reserve

DATASET = "hellaswag"
SPLIT = "validation"
_BRACKETED = re.compile(r"\[.*?\]")


def preprocess(text: str) -> str:
    """The harness's canonical HellaSwag cleanup.

    Reproduced rather than invented: HellaSwag's raw text carries WikiHow
    markup ("[title]", "[step]", "[header]") that no model is meant to predict,
    and every published number applies exactly this normalization. Deviating
    here would make our accuracies incomparable to anyone else's, which is most
    of the point of using a standard benchmark.
    """
    text = text.strip().replace(" [title]", ". ")
    return _BRACKETED.sub("", text).replace("  ", " ")


def load_docs(limit: int = 0) -> list[dict]:
    """(query, four choices, gold) per validation example."""
    from datasets import load_dataset

    ds = load_dataset(DATASET, split=SPLIT)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    docs = []
    for r in ds:
        ctx = r["ctx_a"] + " " + r["ctx_b"].capitalize()
        docs.append(
            {
                "query": preprocess(r["activity_label"] + ": " + ctx),
                "choices": [preprocess(e) for e in r["endings"]],
                "gold": int(r["label"]),
            }
        )
    return docs


def encode_docs(arm, docs: list[dict], heads: dict[str, int]) -> list[dict]:
    """Tokenize each (context, ending) pair with the arm's own tokenizer.

    The continuation boundary is ``len(encode(context))`` -- the context and the
    concatenation are encoded separately and the split is taken by length. For a
    BPE-family tokenizer the context's encoding is not guaranteed to be a strict
    prefix of the concatenation's, so a token straddling the join can shift the
    boundary by one. That is the same approximation every published harness
    makes, it is identical in kind for every arm here, and the alternative
    (scoring the context tokens too) would swamp the signal with context.
    """
    out = []
    for d in docs:
        ctx_len = len(arm.encode(d["query"]).stem)
        seqs = []
        for choice in d["choices"]:
            enc = arm.encode(d["query"] + " " + choice)
            seqs.append(
                {
                    "arrays": {
                        name: getattr(enc, name).astype(np.int64) for name in heads
                    },
                    "ctx_len": ctx_len,
                    "cont_bytes": len((" " + choice).encode("utf-8")),
                }
            )
        out.append({"gold": d["gold"], "seqs": seqs})
    return out


def _pad_batch(seqs: list[dict], heads: dict[str, int], device):
    """Right-pad to the batch's longest sequence.

    Padded target positions carry -100 so ``_ce_tokens``'s ``ignore_index``
    zeroes them; padded *input* positions carry 0, whose hidden state is
    computed and then discarded. Left-padding would be wrong here without an
    attention mask -- the frozen body is called with ``inputs_embeds`` only, so
    a pad on the left would sit inside the causal context of real tokens.
    """
    lens = [len(s["arrays"]["stem"]) for s in seqs]
    width = max(lens) - 1  # inputs are seq[:-1], targets seq[1:]
    inputs = {n: torch.zeros(len(seqs), width, dtype=torch.long) for n in heads}
    targets = {n: torch.full((len(seqs), width), -100, dtype=torch.long) for n in heads}
    keep = torch.zeros(len(seqs), width, dtype=torch.bool)
    for i, s in enumerate(seqs):
        n = lens[i] - 1
        for name in heads:
            a = torch.from_numpy(s["arrays"][name])
            inputs[name][i, :n] = a[:-1]
            targets[name][i, :n] = a[1:]
        # Target position j predicts token j+1, so the continuation starts at
        # j = ctx_len - 1.
        keep[i, max(0, s["ctx_len"] - 1) : n] = True
    return (
        {k: v.to(device) for k, v in inputs.items()},
        {k: v.to(device) for k, v in targets.items()},
        keep.to(device),
    )


@torch.no_grad()
def score(model, encoded: list[dict], heads: dict[str, int], device, batch: int = 8):
    """Summed continuation NLL per choice, over every head.

    Every head is summed, exactly as ``train.eval_bpb`` does. Scoring only the
    stem head would let a factorized arm move its case and morphology cost into
    a side head and appear to predict the continuation better than it does.
    """
    nll = np.zeros((len(encoded), 4), dtype=np.float64)
    flat = [(i, c) for i in range(len(encoded)) for c in range(4)]
    device = torch.device(device)
    for start in range(0, len(flat), batch):
        chunk = flat[start : start + batch]
        seqs = [encoded[i]["seqs"][c] for i, c in chunk]
        inputs, targets, keep = _pad_batch(seqs, heads, device)
        # Autocast only where the BPB harness uses it. On CPU (the tests) it is
        # off, so the arithmetic being checked is the arithmetic, not bf16.
        with torch.autocast(
            device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            per_head = model.token_nll(inputs, targets)
        total = sum(per_head.values())
        sums = (total * keep).sum(dim=1)
        for (i, c), v in zip(chunk, sums.tolist(), strict=True):
            nll[i, c] = v
    return nll


def accuracy(nll: np.ndarray, encoded: list[dict]) -> dict:
    gold = np.array([e["gold"] for e in encoded])
    cont_bytes = np.array(
        [[s["cont_bytes"] for s in e["seqs"]] for e in encoded], dtype=np.float64
    )
    return {
        "acc": float((nll.argmin(1) == gold).mean()),
        "acc_norm": float(((nll / cont_bytes).argmin(1) == gold).mean()),
        "n": len(encoded),
    }


def coverage(encoded: list[dict], ids: np.ndarray) -> dict:
    """How much of the benchmark each arm's active vocabulary can even represent.

    Reported instead of enforced -- see the module docstring. A high number here
    does not invalidate the accuracy above it (the mask is off when scoring), it
    says the arm's corpus-derived vocabulary does not cover the task, which is
    exactly the kind of thing a corpus-BPB headline cannot tell you.
    """
    total = outside = 0
    for e in encoded:
        for s in e["seqs"]:
            stem = s["arrays"]["stem"][s["ctx_len"] :]
            total += stem.size
            outside += int((~np.isin(stem, ids)).sum())
    return {
        "continuation_tokens": total,
        "tokens_outside_active": outside,
        "fraction_outside_active": outside / max(1, total),
    }


def build_native(device):
    """The untouched pretrained model, with no checkpoint involved.

    ``orig-frozen`` is ``train=False``, so it never writes one -- and it is the
    arm every other arm is compared against. Reconstructing it from the host
    table is therefore not a convenience path, it is the only way the reference
    exists at all.
    """
    body, host_emb, _tok, d_model = transplant.load_host(HOST_MODEL)
    heads = {"stem": host_emb.shape[0]}
    model = transplant.TransplantLM(body, heads, d_model, tie_embeddings=True).to(
        device
    )
    transplant.native_init(model, host_emb)
    del host_emb
    return (
        model,
        heads,
        {"experiment": "orig-frozen", "arm": HOST_ARM, "init": "native", "steps": 0},
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ckpt", type=Path)
    g.add_argument("--native", action="store_true", help="score the untouched host")
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    ap.add_argument("--arms-dir", type=Path, default=Path("data/arms"))
    ap.add_argument("--tokens-dir", type=Path, default=Path("data/tokens"))
    ap.add_argument("--out", type=Path, default=Path("results/hellaswag"))
    ap.add_argument("--limit", type=int, default=0, help="0 = the whole split")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--reserve-gb", type=float, default=0.0)
    args = ap.parse_args()

    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    _reserve(args.reserve_gb)
    t0 = time.time()

    if args.native:
        model, heads, meta = build_native(device)
    else:
        from experiments.vocab_factorization.eval_transplant import load_checkpoint

        model, meta = load_checkpoint(args.ckpt, device)
        heads = meta["heads"]
    name = meta["experiment"]
    arm = get_arm(meta["arm"], args.arms_dir, args.corpus)
    print(f"[{name}] arm={meta['arm']} heads={heads}", flush=True)

    docs = load_docs(args.limit)
    encoded = encode_docs(arm, docs, heads)
    print(f"[{name}] {len(encoded)} docs encoded ({time.time() - t0:.0f}s)", flush=True)

    ids = active_vocab.load_or_build(
        meta["arm"],
        arm.heads["stem"],
        args.corpus,
        args.arms_dir,
        args.tokens_dir,
        TransplantTrainSpec().train_bytes,
        workers=args.workers,
    )
    cov = coverage(encoded, ids)
    print(
        f"[{name}] outside active vocab: {cov['tokens_outside_active']}/"
        f"{cov['continuation_tokens']} continuation tokens "
        f"({cov['fraction_outside_active']:.2%})",
        flush=True,
    )

    model.set_active_mask(None)  # explicit: the benchmark is scored unmasked
    nll = score(model, encoded, heads, device, args.batch)
    res = accuracy(nll, encoded)

    results = {
        "experiment": name,
        "arm": meta["arm"],
        "init": meta["init"],
        "steps": meta["steps"],
        "benchmark": f"{DATASET}/{SPLIT}",
        "masked": False,
        "checkpoint": None if args.native else str(args.ckpt),
        **res,
        "active_vocab_coverage": cov,
        "eval_seconds": round(time.time() - t0, 1),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{name}.json").write_text(json.dumps(results, indent=2))
    print(
        f"[{name}] acc={res['acc']:.4f} acc_norm={res['acc_norm']:.4f} "
        f"n={res['n']} ({time.time() - t0:.0f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
