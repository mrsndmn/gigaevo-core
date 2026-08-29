"""Does a cheap fitness rung predict the expensive one? (ADR 0002)

The evolution phase would mutate ``token_features`` and select on a fitness
ladder. A completed GigaEvo run in this repo records ~800 unique evaluations,
which at the trustworthy rung is ~173 GPU-hours — 25x this programme's total
spend, committed to whatever the *cheap* rung happens to reward. This module
buys the right to spend that, or the right not to.

**The gate: Spearman rho >= 0.7 of some cheap rung against the expensive one.**
Above it, the evolution phase is unblocked. Below it, the evolution phase is
killed rather than run on a fitness function that does not track the objective.

The rungs, per candidate routing:

``L0``  seconds. Frequency-weighted explained variance of the mean-of-pieces
        vectors under the candidate partition. Stage 0 finding 4 showed a
        Frobenius-family score does *not* predict BPB, so L0 is here expecting
        to fail; measuring that is the point.
``L1``  ~1 min. Step-0 clean BPB through the frozen body under the active-vocab
        mask. Study 2 measured that step-0 BPB discriminates inits, so this is
        the rung with a real prior in its favour.
``L2``  ~7 min. A *short* transplant train, then clean BPB.

**L2 deviates from the pre-registration, deliberately.** The plan specifies a
33M from-scratch run. But ADR 0003 demoted 33M from "a venue for claims" to
"ladder rung L2" precisely because scale changes the answer — study 2 finding 5
showed the 33M robustness headroom is an artifact of scale. Calibrating a cheap
rung against a *different scale* would measure agreement between two things,
neither of which is what evolution optimizes. A short transplant train is the
same scale, the same body, and the same metric as the real objective, differing
only in how long it runs. That is the honest ground truth for this ladder, and
it is cheaper.

**Range restriction is the trap.** Correlating over near-identical candidates
inflates rho — the plan documents this exact failure elsewhere (``inflation``
rewarding a bad baseline). So the candidate set is feature-group ablations
**plus deliberately-bad anchors**: random routing, single-feature, length-only.
Without a known-bad low end, rho is measured over a restricted range and reads
optimistically.

Run one candidate per job::

    python -m experiments.vocab_factorization.calibrate --candidate surface
    python -m experiments.vocab_factorization.calibrate --analyze
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.vocab_factorization import active_vocab, perturb, transplant
from experiments.vocab_factorization.build_arms import get_arm
from experiments.vocab_factorization.configs import (
    CALIBRATION_CANDIDATES_BY_NAME,
    HOST_MODEL,
    CalibrationCandidate,
    TransplantTrainSpec,
)
from experiments.vocab_factorization.prepare import build_text, encode_text, get_tokens
from experiments.vocab_factorization.train import (
    WindowSampler,
    _heads_arrays,
    eval_bpb,
)
from experiments.vocab_factorization.train_transplant import _heads_for, _reserve

SEQ_LEN = 1024


# --- building a candidate's routing ---------------------------------------


def candidate_features(cand: CalibrationCandidate, feats, vocab: int, seed: int):
    """The feature matrix this candidate routes on.

    The bad anchors are constructed here rather than configured, because each is
    a *different kind* of bad and the difference is the point: `random` has no
    signal at all, `length` has real but nearly useless signal, and a
    single-feature router has signal on one axis only. A rho measured without
    all three is measured over a restricted range.
    """
    if cand.kind == "random":
        rng = np.random.default_rng(seed)
        return rng.normal(size=(vocab, 8)).astype(np.float32)
    if cand.kind == "columns":
        return feats.columns(*cand.columns).matrix
    if cand.kind == "groups":
        return feats.select(*cand.groups).matrix
    raise ValueError(f"unknown candidate kind {cand.kind!r}")


# --- the rungs ------------------------------------------------------------


def l0_explained_variance(
    targets: np.ndarray, routing: np.ndarray, weights: np.ndarray
) -> float:
    """Frequency-weighted explained variance of the partition's leaf means.

    The metric stage 0 disproved, reproduced faithfully so the calibration can
    show whether it deserves its reputation on *this* family of partitions.
    """
    w = weights / max(1e-12, weights.sum())
    mean = (targets * w[:, None]).sum(0)
    total = float((w[:, None] * (targets - mean) ** 2).sum())
    resid = 0.0
    for t in range(routing.shape[1]):
        leaves = routing[:, t]
        recon = np.zeros_like(targets)
        for leaf in np.unique(leaves):
            m = leaves == leaf
            ww = w[m]
            if ww.sum() <= 0:
                continue
            recon[m] = (targets[m] * ww[:, None]).sum(0) / ww.sum()
        resid += float((w[:, None] * (targets - recon) ** 2).sum())
    resid /= routing.shape[1]
    return 1.0 - resid / max(1e-12, total)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, without pulling scipy into the job image."""
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


# --- one candidate --------------------------------------------------------


def run_candidate(cand: CalibrationCandidate, args) -> dict:
    from experiments.vocab_factorization import token_features, tree_embed
    from experiments.vocab_factorization.configs import FeatureSpec, TreeSpec

    device = torch.device("cuda")
    tspec = TransplantTrainSpec()
    arm_name = args.arm
    t0 = time.time()

    body, host_emb, host_tok, d_model = transplant.load_host(HOST_MODEL)
    arm = get_arm(arm_name, args.arms_dir, args.corpus)
    heads = _heads_for(arm, host_emb.shape[0])
    vocab = heads["stem"]

    enc, _ = get_tokens(
        arm_name,
        "train",
        tspec.train_bytes,
        args.corpus,
        args.arms_dir,
        args.tokens_dir,
        workers=args.workers,
    )
    ids = active_vocab.load_or_build(
        arm_name,
        arm.heads["stem"],
        args.corpus,
        args.arms_dir,
        args.tokens_dir,
        tspec.train_bytes,
        workers=args.workers,
    )

    model = transplant.TransplantLM(body, heads, d_model, tie_embeddings=True).to(
        device
    )
    transplant.mean_of_pieces_init(model, arm, host_tok, host_emb, verbose=False)
    targets = model.emb["stem"].weight.detach().float().cpu().numpy()

    feats = token_features.extract(
        arm, FeatureSpec(), host_tok=host_tok, stream=enc.stem
    )
    matrix = candidate_features(cand, feats, vocab, args.seed)
    tree_spec = TreeSpec(
        n_trees=args.n_trees, leaves_per_tree=args.leaves, seed=args.seed
    )
    routing = tree_embed.fit_partition(matrix, targets, tree_spec)
    tree = tree_embed.TreeEmbedding(routing, d_model).to(device)
    model.emb["stem"] = tree
    model.out["stem"] = tree
    model.set_active_mask(active_vocab.mask_from_ids(ids, vocab, device))

    counts = np.bincount(enc.stem.astype(np.int64), minlength=vocab).astype(np.float64)
    l0 = l0_explained_variance(targets, routing.table, counts + 1.0)

    # L1 and L2 share one small eval slice: the ladder is about *ranking*
    # candidates, and a slice that ranks them the same way as the full split
    # costs a minute instead of ten.
    valid_text = build_text(args.corpus, "valid", None)
    cut = int(args.eval_mb * 1024 * 1024)
    valid_text = valid_text[:cut]
    valid_text = valid_text[: valid_text.rfind("\n") + 1]
    e = encode_text(
        arm_name,
        args.arms_dir,
        perturb.apply_suite("clean", valid_text, 0),
        workers=args.workers,
    )
    arrays = _heads_arrays(e, heads)
    nbytes = len(valid_text.encode("utf-8"))

    l1 = eval_bpb(model, arrays, nbytes, SEQ_LEN, device, args.eval_batch)["bpb"]

    # L2: the same objective, run briefly. Not a different scale (see docstring).
    sampler = WindowSampler(_heads_arrays(enc, heads), SEQ_LEN, tspec.seed, device)
    params = model.trainable_parameters()
    opt = torch.optim.AdamW(params, lr=tspec.lr, betas=(0.9, 0.95), weight_decay=0.0)
    micro, accum = args.micro_batch, 2
    model.train()
    body.eval()
    for _ in range(args.l2_steps):
        opt.zero_grad(set_to_none=True)
        for _ in range(accum):
            inp, tgt = sampler.batch(micro)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                losses = model(inp, tgt)
            (sum(losses.values()) / (micro * SEQ_LEN * accum)).backward()
        torch.nn.utils.clip_grad_norm_(params, tspec.grad_clip)
        opt.step()
    l2 = eval_bpb(model, arrays, nbytes, SEQ_LEN, device, args.eval_batch)["bpb"]

    return {
        "candidate": cand.name,
        "kind": cand.kind,
        "is_anchor": cand.is_anchor,
        "arm": arm_name,
        "n_features": int(matrix.shape[1]),
        "n_leaves": routing.n_leaves,
        "L0_explained_variance": l0,
        "L1_step0_bpb": l1,
        "L2_short_train_bpb": l2,
        "l2_steps": args.l2_steps,
        "eval_mb": args.eval_mb,
        "seconds": round(time.time() - t0, 1),
    }


# --- analysis -------------------------------------------------------------


def analyze(results_dir: Path, threshold: float = 0.7) -> dict:
    """Rank-correlate each cheap rung against L2 and return the gate verdict.

    Signs matter and are easy to get backwards: L0 is an explained-variance
    score where *higher* is better, while L1 and L2 are BPB where *lower* is
    better. So a well-behaved L0 correlates **negatively** with L2. Both are
    reported as "agreement with the objective", positive-is-good, with the sign
    flipped once here rather than in every reader's head.
    """
    rows = [json.loads(p.read_text()) for p in sorted(results_dir.glob("*.json"))]
    rows = [r for r in rows if "L2_short_train_bpb" in r]
    if len(rows) < 4:
        return {"error": f"only {len(rows)} candidates present"}
    l0 = np.array([r["L0_explained_variance"] for r in rows])
    l1 = np.array([r["L1_step0_bpb"] for r in rows])
    l2 = np.array([r["L2_short_train_bpb"] for r in rows])

    rho_l0 = -spearman(l0, l2)  # higher EV should mean lower BPB
    rho_l1 = spearman(l1, l2)  # both are BPB
    anchors = [r["candidate"] for r in rows if r["is_anchor"]]
    best = max(rho_l0, rho_l1)
    return {
        "n_candidates": len(rows),
        "n_anchors": len(anchors),
        "anchors": anchors,
        "rho_L0_vs_L2": rho_l0,
        "rho_L1_vs_L2": rho_l1,
        "best_cheap_rung": "L1" if rho_l1 >= rho_l0 else "L0",
        "threshold": threshold,
        "evolution_unblocked": bool(best >= threshold),
        "l2_spread_bpb": float(l2.max() - l2.min()),
        "candidates": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default=None)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--arm", default="bpe8k")
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    ap.add_argument("--arms-dir", type=Path, default=Path("data/arms"))
    ap.add_argument("--tokens-dir", type=Path, default=Path("data/tokens"))
    ap.add_argument("--out", type=Path, default=Path("results/calibration"))
    ap.add_argument("--n-trees", type=int, default=16)
    ap.add_argument("--leaves", type=int, default=124)
    ap.add_argument("--l2-steps", type=int, default=60)
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--eval-mb", type=float, default=0.5)
    ap.add_argument("--eval-batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reserve-gb", type=float, default=0.0)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if args.analyze:
        verdict = analyze(args.out)
        (args.out / "_verdict.json").write_text(json.dumps(verdict, indent=2))
        print(
            json.dumps(
                {k: v for k, v in verdict.items() if k != "candidates"}, indent=2
            )
        )
        return

    if not args.candidate:
        raise SystemExit("pass --candidate NAME or --analyze")
    cand = CALIBRATION_CANDIDATES_BY_NAME[args.candidate]
    torch.set_float32_matmul_precision("high")
    _reserve(args.reserve_gb)
    res = run_candidate(cand, args)
    (args.out / f"{cand.name}.json").write_text(json.dumps(res, indent=2))
    print(
        f"[{cand.name}] L0={res['L0_explained_variance']:.4f} "
        f"L1={res['L1_step0_bpb']:.4f} L2={res['L2_short_train_bpb']:.4f} "
        f"({res['seconds']:.0f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
