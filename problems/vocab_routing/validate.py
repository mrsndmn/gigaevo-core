"""Score an evolved routing-feature program.

The fitness signal is BPB of a routed transplant, so a single evaluation is a
GPU job — which is exactly the fact this problem's existence is gated on. The
ladder rung is chosen by ``VOCAB_ROUTING_RUNG``:

``L0``  seconds, CPU. Frequency-weighted explained variance of the partition.
``L1``  ~1 min, GPU. Step-0 masked BPB through the frozen body.
``L2``  ~7 min, GPU. Short transplant train, then masked BPB.

**No rung is trustworthy until ``calibrate`` says so.** Track 3 measures the
rank correlation of L0 and L1 against L2 over 16 candidates including four
deliberately-bad anchors; the pre-registered gate is Spearman rho >= 0.7, and
below it the evolution phase is killed rather than run. This validator will
happily return an L0 score, and that number means nothing until the gate has
passed. The default is therefore L2 — the expensive, honest one — so that a
run started without reading any of this is slow rather than wrong.

This study has been misled by a cheap proxy twice (study 3's methodological
correction, and stage 0's Frobenius fit), which is why the default is not the
convenient one.
"""

from __future__ import annotations

import os
from pathlib import Path
import time

import numpy as np

RUNG = os.environ.get("VOCAB_ROUTING_RUNG", "L2").upper()
ARM = os.environ.get("VOCAB_ROUTING_ARM", "bpe8k")
DATA = Path(os.environ.get("VOCAB_ROUTING_DATA", "data"))

# The flat control and the masked native reference, both measured (W1).
FLAT_CONTROL_BPB = 0.7704
MASKED_NATIVE_BPB = 0.7796
NOISE_FLOOR_BPB = 0.0016


def _invalid(reason: str) -> dict[str, float]:
    return {
        "fitness": -3.0,
        "is_valid": 0.0,
        "bpb": 3.0,
        "compression": 1.0,
        "n_features": 0.0,
        "feature_seconds": 0.0,
        "error": reason,
    }


def _check(matrix, names, vocab: int) -> str | None:
    """Everything that makes a program unusable rather than merely bad."""
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        return "extract must return a 2-D numpy array"
    if matrix.shape[0] != vocab:
        return f"matrix has {matrix.shape[0]} rows, vocabulary is {vocab}"
    if matrix.shape[1] < 1:
        return "matrix has no columns"
    if len(names) != matrix.shape[1]:
        return f"{len(names)} names for {matrix.shape[1]} columns"
    if not np.isfinite(matrix).all():
        return "matrix contains NaN or inf"
    return None


def validate(program_entrypoint) -> dict[str, float]:
    from experiments.vocab_factorization.build_arms import get_arm
    from experiments.vocab_factorization.configs import TransplantTrainSpec
    from experiments.vocab_factorization.prepare import get_tokens

    if not callable(program_entrypoint):
        return _invalid("entrypoint() must return a callable")
    extract = program_entrypoint
    arm = get_arm(ARM, DATA / "arms", DATA / "corpus")
    vocab = arm.heads["stem"]

    host_tok = None
    if RUNG != "L0":
        from transformers import AutoTokenizer

        from experiments.vocab_factorization.configs import HOST_MODEL

        host_tok = AutoTokenizer.from_pretrained(HOST_MODEL)

    enc, _ = get_tokens(
        ARM,
        "train",
        TransplantTrainSpec().train_bytes,
        DATA / "corpus",
        DATA / "arms",
        DATA / "tokens",
    )

    t0 = time.time()
    try:
        out = extract(arm, host_tok, enc.stem)
    except Exception as exc:  # a program that raises is invalid, not fatal
        return _invalid(f"{type(exc).__name__}: {exc}")
    feature_seconds = time.time() - t0
    if not isinstance(out, tuple) or len(out) != 2:
        return _invalid("extract must return (matrix, names)")
    matrix, names = out
    bad = _check(matrix, names, vocab)
    if bad:
        return _invalid(bad)

    # Determinism is a hard requirement, and cheap to check: a second call must
    # give the same matrix. Without this a program can hill-climb on noise.
    try:
        again, _ = extract(arm, host_tok, enc.stem)
    except Exception as exc:
        return _invalid(f"second call raised {type(exc).__name__}: {exc}")
    if not np.array_equal(np.asarray(matrix), np.asarray(again)):
        return _invalid("extract is not deterministic across calls")

    from experiments.vocab_factorization.scoring import score_routing

    try:
        scored = score_routing(
            matrix.astype(np.float32), arm, host_tok, enc, rung=RUNG, data=DATA
        )
    except Exception as exc:
        return _invalid(f"scoring failed: {type(exc).__name__}: {exc}")

    return {
        "fitness": -scored["bpb"],
        "is_valid": 1.0,
        "bpb": scored["bpb"],
        "compression": scored["compression"],
        "n_features": float(matrix.shape[1]),
        "feature_seconds": round(feature_seconds, 2),
        "rung": RUNG,
    }


def score_on_test(program_entrypoint) -> dict[str, float]:
    """End-of-evolution reporting: the full transplant, never a search signal."""
    prev = os.environ.get("VOCAB_ROUTING_RUNG")
    os.environ["VOCAB_ROUTING_RUNG"] = "L3"
    try:
        res = validate(program_entrypoint)
    finally:
        if prev is None:
            os.environ.pop("VOCAB_ROUTING_RUNG", None)
        else:
            os.environ["VOCAB_ROUTING_RUNG"] = prev
    return {
        "test_bpb": res.get("bpb", 3.0),
        "test_compression": res.get("compression", 1.0),
        "vs_flat_control": res.get("bpb", 3.0) - FLAT_CONTROL_BPB,
    }
