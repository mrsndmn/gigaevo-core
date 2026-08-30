"""Score a saved transplant checkpoint without re-training it.

This is the other half of the checkpointing change, and the reason it was worth
making: study 2 kept only metric JSONs, so every later question about those five
models — do they still follow instructions? how do they score on a downstream
task? — required a full re-train before it could even be asked. With the
embedding and heads on disk, a new evaluation costs the evaluation.

The body is not in the checkpoint and does not need to be: it was frozen, so it
reloads from HF bit-identical. What is restored is exactly what changed.

Run::

    python -m experiments.vocab_factorization.eval_transplant --ckpt results/transplant/ckpt/bpe8k-t.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from experiments.vocab_factorization import active_vocab, perturb, transplant
from experiments.vocab_factorization.build_arms import get_arm
from experiments.vocab_factorization.configs import (
    HOST_MODEL,
    TransplantTrainSpec,
)
from experiments.vocab_factorization.prepare import build_text, encode_text
from experiments.vocab_factorization.train import EVAL_SUITES, _heads_arrays, eval_bpb
from experiments.vocab_factorization.train_transplant import _reserve

SEQ_LEN = 1024


def load_checkpoint(path: Path, device) -> tuple[torch.nn.Module, dict]:
    """Rebuild the model a checkpoint describes. Returns (model, metadata)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    # Both checks run before ``load_host``: loading a 1.7B model takes the best
    # part of a minute, and a checkpoint that cannot be reconstructed should say
    # so immediately rather than after it.
    if ckpt.get("host") != HOST_MODEL:
        raise ValueError(
            f"{path} was trained against host {ckpt.get('host')!r}, not "
            f"{HOST_MODEL!r} — the frozen body would not match the embedding"
        )
    frozen = ckpt.get("freeze_body", True)
    if not frozen and "body" not in ckpt:
        # An arm that trained the body must restore it. Falling back to the HF
        # weights here would pair a pristine body with an embedding trained
        # against a different one and report the result as that arm's score.
        raise ValueError(
            f"{path} is an unfrozen-body arm but carries no body weights — "
            "it was written by a version that did not save them, and the "
            "model it describes cannot be reconstructed"
        )
    body, host_emb, _tok, d_model = transplant.load_host(ckpt["host"])
    model = transplant.TransplantLM(
        body, ckpt["heads"], d_model, tie_embeddings=True, freeze_body=frozen
    ).to(device)
    model.emb.load_state_dict(ckpt["emb"])
    # The tied arm's ``out.stem`` shares storage with ``emb.stem``; loading it
    # again is a no-op there and load-bearing for an untied one.
    model.out.load_state_dict(ckpt["out"])
    if not frozen:
        model.body.load_state_dict(ckpt["body"])
    meta = {k: v for k, v in ckpt.items() if k not in ("emb", "out", "body")}
    # A shrink arm's ``heads`` describe the *rewritten* table, so the host's own
    # row count is no longer recoverable from the checkpoint — and rebuilding its
    # codec needs it (``shrink.host_vocabulary`` pads to exactly this many rows).
    # It is free here and a second 1.7B load anywhere else.
    meta["host_vocab"] = int(host_emb.shape[0])
    del host_emb
    return model, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    ap.add_argument("--arms-dir", type=Path, default=Path("data/arms"))
    ap.add_argument("--tokens-dir", type=Path, default=Path("data/tokens"))
    ap.add_argument("--out", type=Path, default=Path("results/transplant"))
    ap.add_argument("--eval-batch", type=int, default=4)
    ap.add_argument("--eval-mb", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--reserve-gb", type=float, default=0.0)
    ap.add_argument("--no-mask", action="store_true")
    args = ap.parse_args()

    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    _reserve(args.reserve_gb)

    t0 = time.time()
    model, meta = load_checkpoint(args.ckpt, device)
    name = meta["experiment"]
    arm_name = meta["arm"]
    print(
        f"[{name}] loaded {args.ckpt} (arm={arm_name} init={meta['init']} "
        f"steps={meta['steps']})",
        flush=True,
    )

    # ``run_eval`` enumerates every ``*.pt`` under the artifacts root, and the
    # shrink wave put nine more there. This module encodes its eval suites with
    # ``arm.encode``, which yields *host* ids — scoring those against a rewritten
    # table would not crash, it would index rows that mean something else and
    # report the result as that arm's BPB. Refusing is the only honest option
    # until this path grows the codec rebuild ``hellaswag`` has.
    if meta.get("stage", 0):
        raise SystemExit(
            f"{args.ckpt} is a progressive-shrink checkpoint (stage "
            f"{meta['stage']}, curriculum {meta.get('curriculum', 'default')}) "
            "whose model indexes a rewritten vocabulary. Its BPB is already in "
            f"results/transplant/{name}.json, written by the run that trained it "
            "with the codec applied to the eval streams."
        )

    arm = get_arm(arm_name, args.arms_dir, args.corpus)
    heads = meta["heads"]

    valid_text = build_text(args.corpus, "valid", None)
    if args.eval_mb:
        cut = int(args.eval_mb * 1024 * 1024)
        valid_text = valid_text[:cut]
        valid_text = valid_text[: valid_text.rfind("\n") + 1]

    eval_sets = {}
    for suite in EVAL_SUITES:
        text = perturb.apply_suite(suite, valid_text, seed=0)
        e = encode_text(arm_name, args.arms_dir, text, workers=args.workers)
        eval_sets[suite] = (_heads_arrays(e, heads), len(text.encode("utf-8")))

    mask_info = None
    if not args.no_mask:
        ids = active_vocab.load_or_build(
            arm_name,
            arm.heads["stem"],
            args.corpus,
            args.arms_dir,
            args.tokens_dir,
            TransplantTrainSpec().train_bytes,
            workers=args.workers,
        )
        for suite in EVAL_SUITES:
            active_vocab.assert_targets_active(
                eval_sets[suite][0], ids, f"{name}/{suite}"
            )
        model.set_active_mask(active_vocab.mask_from_ids(ids, heads["stem"], device))
        mask_info = {"n_active": int(ids.size), "table_size": heads["stem"]}

    results: dict = {
        "experiment": name,
        "arm": arm_name,
        "host": meta["host"],
        "init": meta["init"],
        "steps": meta["steps"],
        "checkpoint": str(args.ckpt),
        "active_vocab": mask_info,
        "source": "eval_transplant",
    }
    for suite in EVAL_SUITES:
        arrays, nb = eval_sets[suite]
        m = eval_bpb(model, arrays, nb, SEQ_LEN, device, args.eval_batch)
        results[f"eval.{suite}"] = m
        print(f"[{name}] {suite:9s} bpb={m['bpb']:.4f}", flush=True)
    clean = results["eval.clean"]["bpb"]
    for suite in EVAL_SUITES:
        results[f"delta.{suite}"] = results[f"eval.{suite}"]["bpb"] - clean
    results["eval_seconds"] = round(time.time() - t0, 1)

    # A reloaded checkpoint must reproduce the run that saved it. Where the
    # original run's record is at hand, say so in the artifact rather than
    # leaving the comparison to whoever reads it next.
    original = args.out / f"{name}.json"
    if original.exists():
        prev = json.loads(original.read_text()).get("eval.clean", {}).get("bpb")
        if prev is not None:
            results["reload_delta_clean"] = clean - prev
            print(
                f"[{name}] reload check: {clean:.4f} vs {prev:.4f} recorded "
                f"({clean - prev:+.4f})",
                flush=True,
            )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{name}.eval.json").write_text(json.dumps(results, indent=2))
    print(f"[{name}] -> {args.out / f'{name}.eval.json'}", flush=True)


if __name__ == "__main__":
    main()
