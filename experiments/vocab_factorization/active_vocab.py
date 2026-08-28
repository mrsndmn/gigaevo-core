"""The active vocabulary: which host ids this corpus actually exercises.

Stage 0 measured that only ~72k of Qwen3's 151,936 rows occur in the 230 MB
ru+en train stream; the rest is mostly other-language mass this corpus never
touches. Those rows still sit in the softmax normalizer, where they cost every
arm a different amount depending on how its embeddings happen to land near
them — so per-arm deltas measured without a shared mask are not comparable.

The convention (plan §"Active-vocabulary convention"), applied to **every** arm
including the native reference:

(a) *active* = ids occurring in the train stream or the valid split;
(b) only active tokens are parameterized and optimized;
(c) inactive tokens are **masked out of the softmax in training and
    evaluation**;
(d) compression ratios are quoted against the *active* flat table
    (``n_active x d_model``), not the full one.

(b) needs no special code: an id that never appears as an input gets no
embedding gradient, and (c) drives its output-row gradient to exactly zero
through the softmax, so masking *is* the restriction. ``active_floats``
implements (d).

**Which texts define "the valid split".** The pre-registered wording says the
clean valid split, which was written when only clean BPB was in view. The
harness also scores six perturbation suites, and a perturbed text emits ids the
clean text never did — under (c) those become impossible targets worth infinite
BPB, which would take the robustness suites (a stage-1 gate) off the table. The
set is therefore built over **every text the harness scores**: train + clean +
the perturbation suites. The cost of the extension is recorded rather than
assumed — ``--report`` writes the per-suite leakage into the cache.

Measured on Qwen3-1.7B over this corpus (``--report``, 2026-08-28):

===============================  =======  ======================================
train stream (230 MB)             72,304  the stage-0 number, reproduced exactly
+ clean valid split               72,477  the *pre-registered* set
+ the six perturbation suites     74,957  what the harness actually needs
full table                       151,936  (151,669 tokenizer + 267 padding rows)
===============================  =======  ======================================

The extension is not cosmetic: the ``upper`` suite alone emits 1,323 ids outside
the pre-registered set, covering **3.4% of its tokens**. Masking those would not
have shifted ``upper`` BPB slightly — it would have made it unbounded. The
active flat table is therefore 74,957 x 2048 = **153.5M floats**, the number
compression ratios are quoted against, in place of the plan's 148.1M.

Run::

    python -m experiments.vocab_factorization.active_vocab --report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.vocab_factorization import perturb
from experiments.vocab_factorization.configs import (
    HOST_ARM,
    TransplantTrainSpec,
    active_cache_path,
)
from experiments.vocab_factorization.prepare import (
    build_text,
    encode_text,
    get_tokens,
)
from experiments.vocab_factorization.train import EVAL_SUITES

# The texts an id may appear in to count as active. "clean" is the valid split
# itself; the rest are the perturbation suites the BPB harness scores.
ACTIVE_SUITES: tuple[str, ...] = EVAL_SUITES

active_path = active_cache_path


def _unique(a: np.ndarray) -> np.ndarray:
    return np.unique(a.astype(np.int64))


def build(
    arm_name: str,
    vocab_size: int,
    corpus_dir: Path,
    arms_dir: Path,
    tokens_dir: Path,
    train_bytes: int,
    workers: int = 40,
    verbose: bool = True,
) -> dict:
    """Compute the active id set and the audit numbers behind it.

    Returns a dict with ``ids`` (sorted int64) plus per-source counts. The
    train stream and the clean valid split come from the shared token cache;
    the perturbation suites are encoded here because they are functions of the
    valid text rather than of a split.
    """
    enc_train, _ = get_tokens(
        arm_name, "train", train_bytes, corpus_dir, arms_dir, tokens_dir, workers
    )
    train_ids = _unique(enc_train.stem)
    del enc_train

    enc_valid, _ = get_tokens(
        arm_name, "valid", None, corpus_dir, arms_dir, tokens_dir, workers
    )
    clean_ids = _unique(enc_valid.stem)
    del enc_valid

    # The pre-registered set: train + the clean valid split.
    registered = np.union1d(train_ids, clean_ids)

    valid_text = build_text(corpus_dir, "valid", None)
    per_suite: dict[str, dict] = {}
    ids = registered
    for suite in ACTIVE_SUITES:
        if suite == "clean":
            suite_ids, n_tok = clean_ids, None
        else:
            e = encode_text(
                arm_name,
                arms_dir,
                perturb.apply_suite(suite, valid_text, seed=0),
                workers,
            )
            suite_ids, n_tok = _unique(e.stem), len(e.stem)
            outside = suite_ids[~np.isin(suite_ids, registered)]
            leaked = int(np.isin(e.stem.astype(np.int64), outside).sum())
            per_suite[suite] = {
                "ids": int(suite_ids.size),
                "ids_outside_registered": int(outside.size),
                "tokens_outside_registered": leaked,
                "token_fraction_outside": leaked / max(1, n_tok),
            }
            del e
        ids = np.union1d(ids, suite_ids)

    return {
        "arm": arm_name,
        "vocab_size": int(vocab_size),
        "train_bytes": int(train_bytes),
        "suites": list(ACTIVE_SUITES),
        "n_train_ids": int(train_ids.size),
        "n_clean_ids": int(clean_ids.size),
        "n_registered": int(registered.size),
        "n_active": int(ids.size),
        "coverage": float(ids.size / vocab_size),
        "per_suite": per_suite,
        "ids": ids,
    }


def load_or_build(
    arm_name: str,
    vocab_size: int,
    corpus_dir: Path,
    arms_dir: Path,
    tokens_dir: Path,
    train_bytes: int | None = None,
    workers: int = 40,
    verbose: bool = True,
) -> np.ndarray:
    """Return the cached active id set, computing it on first use.

    The cache is keyed by arm only, and records the vocabulary size and train
    budget it was built under: a mismatch means the convention would silently
    change between arms, which is the one thing the mask exists to prevent.
    """
    train_bytes = (
        TransplantTrainSpec().train_bytes if train_bytes is None else train_bytes
    )
    path = active_path(tokens_dir, arm_name)
    if path.exists():
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        if meta["vocab_size"] != vocab_size or meta["train_bytes"] != train_bytes:
            raise ValueError(
                f"{path} was built for vocab_size={meta['vocab_size']} "
                f"train_bytes={meta['train_bytes']}, not "
                f"{vocab_size}/{train_bytes} — delete it to rebuild"
            )
        ids = z["ids"]
        if verbose:
            print(
                f"active vocab: {ids.size}/{vocab_size} ids "
                f"({ids.size / vocab_size:.1%}) [cached]",
                flush=True,
            )
        return ids

    info = build(
        arm_name, vocab_size, corpus_dir, arms_dir, tokens_dir, train_bytes, workers
    )
    ids = info.pop("ids")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ids=ids, meta=json.dumps(info))
    if verbose:
        print(
            f"active vocab: {ids.size}/{vocab_size} ids "
            f"({ids.size / vocab_size:.1%}) -> {path}",
            flush=True,
        )
    return ids


def mask_from_ids(ids: np.ndarray, table_size: int, device=None):
    """Boolean keep-mask over the embedding *table*: True where an id is active.

    ``table_size`` is the model's row count, which for an HF host is the padded
    151,936 rather than ``len(tokenizer)`` = 151,669 (see
    ``train_transplant._heads_for``). The padding rows are unreachable by the
    tokenizer, so they are inactive by construction and the mask removes them
    from the normalizer along with the unused-language mass.
    """
    import torch

    mask = torch.zeros(table_size, dtype=torch.bool)
    mask[torch.from_numpy(np.asarray(ids, dtype=np.int64))] = True
    return mask if device is None else mask.to(device)


def assert_targets_active(
    arrays: dict[str, np.ndarray], ids: np.ndarray, what: str
) -> None:
    """Fail loudly if a text scores tokens the mask makes impossible to emit.

    A target outside the active set has probability zero under the mask, so its
    contribution to BPB is unbounded. Catching it here turns a silently absurd
    number into an error naming the text that caused it.
    """
    stem = arrays["stem"]
    bad = np.setdiff1d(np.unique(stem.astype(np.int64)), ids, assume_unique=True)
    if bad.size:
        n = int(np.isin(stem.astype(np.int64), bad).sum())
        raise ValueError(
            f"{what}: {bad.size} of its token ids are outside the active "
            f"vocabulary ({n} tokens, {n / stem.size:.4%}), e.g. "
            f"{bad[:8].tolist()} — they would be scored at probability zero. "
            "Rebuild the active set over every text the harness scores "
            "(python -m experiments.vocab_factorization.active_vocab --rebuild)."
        )


def active_floats(n_active: int, d_model: int) -> int:
    """Convention (d): the flat table a routed one is compressed against."""
    return n_active * d_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default=HOST_ARM)
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    ap.add_argument("--arms-dir", type=Path, default=Path("data/arms"))
    ap.add_argument("--tokens-dir", type=Path, default=Path("data/tokens"))
    ap.add_argument("--train-mb", type=int, default=0)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--report", action="store_true", help="print the audit and exit")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    from experiments.vocab_factorization.build_arms import get_arm

    arm = get_arm(args.arm, args.arms_dir, args.corpus)
    vocab = arm.heads["stem"]
    train_bytes = (
        args.train_mb * 1024 * 1024
        if args.train_mb
        else TransplantTrainSpec().train_bytes
    )

    path = active_path(args.tokens_dir, args.arm)
    if args.rebuild and path.exists():
        path.unlink()

    if args.report:
        info = build(
            args.arm,
            vocab,
            args.corpus,
            args.arms_dir,
            args.tokens_dir,
            train_bytes,
            args.workers,
        )
        ids = info.pop("ids")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, ids=ids, meta=json.dumps(info))
        print(json.dumps(info, indent=2))
        print(f"\nwrote {path}")
        d_model = 2048
        print(
            f"active flat table: {info['n_active']} x {d_model} = "
            f"{active_floats(info['n_active'], d_model) / 1e6:.1f}M floats "
            f"(full: {vocab * d_model / 1e6:.1f}M)"
        )
        return

    load_or_build(
        args.arm,
        vocab,
        args.corpus,
        args.arms_dir,
        args.tokens_dir,
        train_bytes,
        args.workers,
    )


if __name__ == "__main__":
    main()
