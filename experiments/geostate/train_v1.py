"""V1: LoRA-SFT of Qwen3-1.7B on generator trajectories, two arms.

    python -m experiments.geostate.train_v1 --arm tokens --seed 1 --out DIR --train --eval
    python -m experiments.geostate.train_v1 --arm table  --seed 1 --out DIR --train --eval --smoke

Data (AC-15): trajectories from the programmatic generator, one seed for
training and another for evaluation; evaluation problems that coincide with a
training problem are dropped. Both arms train on the same problems, the same
LoRA, the same optimiser and the same targets — the move lines — and differ
only in the state blocks (AC-16). `--smoke` runs a few steps on a few problems
to exercise the whole path on one GPU.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .client import read_json, write_json
from .configs import (
    ARM_TABLE,
    ARM_TOKENS,
    ARTIFACT_ROOT,
    GENERATOR,
    OUT_DIR,
    TOL,
    V1,
    ToleranceSpec,
    V1Config,
)
from .encoder import pad_features
from .features import object_features
from .generator import Problem, generate_with_stats, load_problems, save_problems
from .sequence import build_sequence

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def data_key(cfg: V1Config, smoke: bool = False) -> str:
    """The data depends on the generator and kernel code as much as on seeds."""
    import inspect

    from . import cga, generator, macros, state

    material = json.dumps(
        {
            "code": [inspect.getsource(m) for m in (cga, state, macros, generator)],
            "train_seed": cfg.train_seed,
            "train_per_bucket": cfg.smoke_train_per_bucket
            if smoke
            else cfg.train_per_bucket,
            "eval_seed": cfg.eval_seed,
            "eval_per_bucket": cfg.smoke_eval_per_bucket
            if smoke
            else cfg.eval_per_bucket,
            "generator": asdict(GENERATOR),
            "tolerances": asdict(TOL),
        },
        sort_keys=True,
    )
    return hashlib.sha256(material.encode()).hexdigest()[:10]


def data_dir(
    cfg: V1Config, root: str | Path | None = None, smoke: bool = False
) -> Path:
    base = (
        Path(root)
        if root is not None
        else (OUT_DIR / "v1" if smoke else Path(ARTIFACT_ROOT) / "v1")
    )
    return base / f"data-{data_key(cfg, smoke)}"


def _key(p: Problem) -> str:
    return json.dumps({"given": p.to_dict()["given"], "expr": p.expr}, sort_keys=True)


def prepare_data(
    cfg: V1Config,
    root: str | Path | None = None,
    smoke: bool = False,
    tol: ToleranceSpec = TOL,
) -> tuple[Path, Path]:
    """Generate (once) the train and eval problem files; concurrent jobs share them."""
    d = data_dir(cfg, root, smoke)
    train_path, eval_path = d / "train.jsonl", d / "eval.jsonl"
    manifest = d / "data.json"  # written last: its presence means "complete"
    if manifest.exists():
        return train_path, eval_path
    d.mkdir(parents=True, exist_ok=True)
    with (d / "prepare.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if manifest.exists():
            return train_path, eval_path
        per_train = cfg.smoke_train_per_bucket if smoke else cfg.train_per_bucket
        per_eval = cfg.smoke_eval_per_bucket if smoke else cfg.eval_per_bucket
        started = time.time()
        train, train_stats = generate_with_stats(
            GENERATOR, tol, cfg.train_seed, per_train, prefix="train"
        )
        evaluation, eval_stats = generate_with_stats(
            GENERATOR, tol, cfg.eval_seed, per_eval, prefix="eval"
        )
        seen = {_key(p) for p in train}
        kept = [p for p in evaluation if _key(p) not in seen]
        save_problems(train, train_path)
        save_problems(kept, eval_path)
        write_json(
            d / "data.json",
            {
                "key": data_key(cfg, smoke),
                "n_train": len(train),
                "n_eval": len(kept),
                "eval_dropped_as_seen": len(evaluation) - len(kept),
                "train_stats": train_stats.to_dict(),
                "eval_stats": eval_stats.to_dict(),
                "seconds": time.time() - started,
            },
        )
    return train_path, eval_path


class TrajectoryDataset(Dataset):
    """Teacher-forced sequences with loss on move lines only."""

    def __init__(
        self,
        problems: list[Problem],
        arm: str,
        tok,
        cfg: V1Config,
        tol: ToleranceSpec = TOL,
    ):
        self.problems = problems
        self.arm = arm
        self.tok = tok
        self.cfg = cfg
        self.tol = tol
        # The keep set is decided on the *table* arm's lengths (the longer of
        # the two) for both arms, so both train on the same problems in the
        # same order whatever the arm (AC-16).
        texts = [build_sequence(p, ARM_TABLE)[0].text for p in problems]
        lengths = [
            len(ids) for ids in tok(texts, add_special_tokens=False)["input_ids"]
        ]
        self.keep = [i for i, n in enumerate(lengths) if n <= cfg.max_len]
        self.dropped = len(problems) - len(self.keep)

    def __len__(self) -> int:
        return len(self.keep)

    def encode(self, problem: Problem) -> dict[str, Any]:
        seq, state = build_sequence(problem, self.arm)
        ids: list[int] = []
        labels: list[int] = []
        for seg in seq.segments:
            seg_ids = self.tok(seg.text, add_special_tokens=False)["input_ids"]
            ids.extend(seg_ids)
            labels.extend(seg_ids if seg.loss else [-100] * len(seg_ids))
        feats = []
        if self.arm == ARM_TOKENS:
            for h in seq.state_handles:
                s, p, r = object_features(state, h)
                feats.append(
                    (torch.from_numpy(s), torch.from_numpy(p), torch.from_numpy(r))
                )
        return {"input_ids": ids, "labels": labels, "features": feats}

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.encode(self.problems[self.keep[index]])


def collate(batch: list[dict[str, Any]], pad_id: int) -> dict[str, torch.Tensor]:
    n = max(len(b["input_ids"]) for b in batch)
    input_ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), n), -100, dtype=torch.long)
    attention = torch.zeros((len(batch), n), dtype=torch.long)
    feats = []
    for i, b in enumerate(batch):
        m = len(b["input_ids"])
        input_ids[i, :m] = torch.tensor(b["input_ids"])
        labels[i, :m] = torch.tensor(b["labels"])
        attention[i, :m] = 1
        feats.extend(b["features"])
    self_feats, pair_feats, pair_mask, pair_index = pad_features(feats)
    return {
        "input_ids": input_ids,
        "attention_mask": attention,
        "labels": labels,
        "self_feats": self_feats,
        "pair_feats": pair_feats,
        "pair_mask": pair_mask,
        "pair_index": pair_index,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def lr_schedule(warmup: int, total: int):
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return f


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(
    model,
    problems: list[Problem],
    cfg: V1Config,
    seed: int,
    out: Path,
    smoke: bool = False,
    device: str = "cuda",
) -> dict[str, Any]:
    seed_everything(seed)
    out.mkdir(parents=True, exist_ok=True)
    stale = out / "eval.jsonl"
    if stale.exists():
        # episodes of a previous model must never be resumed under a new one
        stale.rename(out / f"eval.stale-{int(time.time())}.jsonl")
    dataset = TrajectoryDataset(problems, model.arm, model.tok, cfg)
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, model.pad_id),
        num_workers=2,
        generator=gen,
    )
    steps_per_epoch = math.ceil(len(loader) / cfg.grad_accum)
    total_steps = cfg.smoke_steps if smoke else cfg.epochs * steps_per_epoch
    optimizer = torch.optim.AdamW(model.param_groups(), weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_schedule(min(cfg.warmup_steps, total_steps // 10 + 1), total_steps),
    )
    print(
        f"[{model.arm} s{seed}] {len(dataset)} sequences ({dataset.dropped} dropped for "
        f"length), {total_steps} optimiser steps, {model.n_trainable():,} trainable params",
        flush=True,
    )
    model.train()
    history: list[dict[str, float]] = []
    step = 0
    micro = 0
    started = time.monotonic()
    running = 0.0
    done = False
    for epoch in range(cfg.epochs if not smoke else 1000):
        for batch in loader:
            with torch.autocast(
                device_type="cuda" if "cuda" in str(device) else "cpu",
                dtype=torch.bfloat16,
            ):
                loss = model(**batch)
            (loss / cfg.grad_accum).backward()
            running += float(loss.item())
            micro += 1
            if micro % cfg.grad_accum == 0:
                # clip each group on its own: the LoRA step must not depend on
                # the encoder's gradient norm (the table arm has no encoder)
                for group in model.param_groups():
                    torch.nn.utils.clip_grad_norm_(group["params"], cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % cfg.log_every == 0 or step == total_steps or smoke:
                    avg = running / (
                        cfg.grad_accum * (cfg.log_every if not smoke else 1)
                    )
                    history.append(
                        {"step": step, "loss": avg, "lr": scheduler.get_last_lr()[0]}
                    )
                    print(
                        f"  step {step}/{total_steps} loss {avg:.4f} "
                        f"({time.monotonic() - started:.0f}s)",
                        flush=True,
                    )
                    running = 0.0
                if step >= total_steps:
                    done = True
                    break
        if done:
            break
    adapters = model.save_adapters(out / "adapters.pt")
    report = {
        "arm": model.arm,
        "seed": seed,
        "steps": step,
        "n_sequences": len(dataset),
        "dropped_for_length": dataset.dropped,
        "trainable_params": model.n_trainable(),
        "history": history,
        "seconds": time.monotonic() - started,
        "adapters": str(adapters),
        "smoke": smoke,
    }
    write_json(out / "train.json", report)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=V1.arms, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--retrain",
        action="store_true",
        help="train even if adapters.pt already exists (default: reuse them)",
    )
    args = ap.parse_args(argv)
    cfg = V1
    from .v1_model import V1Model

    # Seed before the model is built: LoRA A and the encoder are initialised
    # there, and both arms of one seed should share the LoRA init.
    seed_everything(args.seed)
    train_path, eval_path = prepare_data(cfg, args.data_root, args.smoke)
    data_manifest = read_json(train_path.parent / "data.json")
    print(f"data: {train_path.parent} ({data_manifest['n_train']} train)", flush=True)
    model = V1Model(cfg, args.arm, device=args.device)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    adapters = out / "adapters.pt"
    trained_before = adapters.exists() and (out / "train.json").exists()
    write_json(
        out / "run.json",
        {
            "arm": args.arm,
            "seed": args.seed,
            "data_dir": str(train_path.parent),
            "data_key": data_manifest["key"],
            "n_train": data_manifest["n_train"],
            "n_eval": data_manifest["n_eval"],
            "smoke": args.smoke,
        },
    )
    if args.train and trained_before and not args.retrain:
        print(
            f"adapters exist at {adapters}; reusing them (pass --retrain to train again)",
            flush=True,
        )
        model.load_adapters(adapters)
    elif args.train:
        train(
            model,
            load_problems(train_path),
            cfg,
            args.seed,
            out,
            smoke=args.smoke,
            device=args.device,
        )
    if args.eval:
        from .rollout import evaluate

        if not args.train:
            model.load_adapters(adapters)
        problems = load_problems(eval_path)
        metrics = evaluate(model, problems, cfg, out)
        print(json.dumps(metrics["buckets"], indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
