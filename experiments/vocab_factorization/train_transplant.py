"""Run the vocabulary-transplant grid on a frozen pretrained body.

Reuses study 1's corpus, tokenizer arms, perturbation suites, and — crucially —
its bits-per-byte evaluator, so the two sets of numbers are directly
comparable in kind (not in value: different body, different corpus budget).

Every arm is evaluated **twice**: once at step 0, before any training, and once
after. The step-0 row under ``mean_of_pieces`` init is the literal
"modify the vocabulary on the fly, train nothing" measurement.

Run::

    python -m experiments.vocab_factorization.train_transplant --all
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import torch

from experiments.vocab_factorization import perturb, transplant
from experiments.vocab_factorization.build_arms import get_arm
from experiments.vocab_factorization.configs import (
    HOST_MODEL,
    TRANSPLANTS,
    TRANSPLANTS_BY_NAME,
    TransplantSpec,
    TransplantTrainSpec,
)
from experiments.vocab_factorization.prepare import build_text, encode_text, get_tokens
from experiments.vocab_factorization.train import (
    EVAL_SUITES,
    WindowSampler,
    _heads_arrays,
    eval_bpb,
)


def _heads_for(arm, host_vocab: int) -> dict[str, int]:
    """Head cardinalities, with the host's padded table size for the native arm.

    ``len(tokenizer)`` is 151,669 for Qwen3 but the pretrained embedding matrix
    is 151,936 rows; using the tokenizer length would silently reshape the
    table we are trying to copy.
    """
    heads = dict(arm.heads)
    if getattr(arm, "kind", "") == "hf":
        heads["stem"] = host_vocab
    return heads


def run(spec: TransplantSpec, host, args, tspec: TransplantTrainSpec) -> dict:
    body, host_emb, host_tok, d_model = host
    device = torch.device("cuda")
    torch.manual_seed(tspec.seed)

    arm = get_arm(spec.arm, args.arms_dir, args.corpus)
    heads = _heads_for(arm, host_emb.shape[0])

    budget = args.train_mb * 1024 * 1024 if args.train_mb else tspec.train_bytes
    enc, train_bytes = get_tokens(
        spec.arm,
        "train",
        budget,
        args.corpus,
        args.arms_dir,
        args.tokens_dir,
        workers=args.workers,
    )
    n_tokens = len(enc)

    model = transplant.TransplantLM(body, heads, d_model, tie_embeddings=True).to(
        device
    )
    if spec.init == "native":
        init_stats = transplant.native_init(model, host_emb)
    elif spec.init == "random":
        init_stats = transplant.random_init(model, host_emb)
    elif spec.init == "mean_of_pieces":
        init_stats = transplant.mean_of_pieces_init(model, arm, host_tok, host_emb)
    else:
        raise ValueError(spec.init)
    counts = model.param_counts()

    seq = tspec.seq_len
    micro = spec.micro_batch
    accum = max(1, tspec.batch_tokens // (micro * seq))
    steps = 0 if not spec.train else (args.steps or int(n_tokens / tspec.batch_tokens))
    print(
        f"[{spec.name}] arm={spec.arm} init={spec.init} heads={heads} "
        f"tokens={n_tokens / 1e6:.1f}M bytes={train_bytes / 1e6:.0f}MB "
        f"steps={steps} micro={micro} accum={accum} params={counts}",
        flush=True,
    )

    valid_text = build_text(args.corpus, "valid", None)
    if args.eval_mb:
        cut = int(args.eval_mb * 1024 * 1024)
        valid_text = valid_text[:cut]
        valid_text = valid_text[: valid_text.rfind("\n") + 1]

    # Encode every suite once and reuse for both the step-0 and final passes:
    # the text does not change, only the model does.
    eval_sets = {}
    for suite in EVAL_SUITES:
        text = perturb.apply_suite(suite, valid_text, seed=0)
        e = encode_text(spec.arm, args.arms_dir, text, workers=args.workers)
        eval_sets[suite] = (_heads_arrays(e, heads), len(text.encode("utf-8")))

    def evaluate(tag: str, net) -> dict:
        out = {}
        for suite in EVAL_SUITES:
            arrays, nb = eval_sets[suite]
            m = eval_bpb(net, arrays, nb, seq, device, args.eval_batch)
            out[suite] = m
            print(f"[{spec.name}] {tag:6s} {suite:9s} bpb={m['bpb']:.4f}", flush=True)
        return out

    # Step 0: the "on the fly, train nothing" measurement.
    pre = evaluate("step0", model)

    train_seconds = 0.0
    if steps:
        warmup = max(1, int(steps * tspec.warmup_frac))
        params = model.trainable_parameters()
        opt = torch.optim.AdamW(
            params, lr=tspec.lr, betas=(0.9, 0.95), weight_decay=tspec.weight_decay
        )

        def lr_at(s: int) -> float:
            if s < warmup:
                return tspec.lr * (s + 1) / warmup
            p = (s - warmup) / max(1, steps - warmup)
            f = tspec.min_lr_frac
            return tspec.lr * (f + (1 - f) * 0.5 * (1 + math.cos(math.pi * p)))

        sampler = WindowSampler(_heads_arrays(enc, heads), seq, tspec.seed, device)
        model.train()
        body.eval()  # frozen body stays in eval mode (no dropout)
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
                tot += loss.detach()
            torch.nn.utils.clip_grad_norm_(params, tspec.grad_clip)
            opt.step()
            if step % tspec.log_every == 0 or step == steps - 1:
                print(
                    f"[{spec.name}] step {step:5d}/{steps} loss/token={float(tot):.4f} "
                    f"lr={lr_at(step):.2e} {time.time() - t0:7.1f}s",
                    flush=True,
                )
        train_seconds = time.time() - t0

    post = evaluate("final", model) if steps else pre

    results: dict = {
        "experiment": spec.name,
        "arm": spec.arm,
        "host": HOST_MODEL,
        "init": spec.init,
        "init_stats": init_stats,
        "heads": heads,
        "params": counts,
        "steps": steps,
        "train_tokens": n_tokens,
        "train_bytes": train_bytes,
        "train_seconds": round(train_seconds, 1),
        "tokens_per_byte_train": n_tokens / train_bytes,
        "context_bytes": seq / (n_tokens / train_bytes),
        "note": spec.note,
    }
    for suite in EVAL_SUITES:
        results[f"eval.{suite}"] = post[suite]
        results[f"step0.{suite}"] = pre[suite]
    clean = post["clean"]["bpb"]
    for suite in EVAL_SUITES:
        results[f"delta.{suite}"] = post[suite]["bpb"] - clean
    results["step0_clean_bpb"] = pre["clean"]["bpb"]

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{spec.name}.json").write_text(json.dumps(results, indent=2))
    del model
    torch.cuda.empty_cache()
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    ap.add_argument("--arms-dir", type=Path, default=Path("data/arms"))
    ap.add_argument("--tokens-dir", type=Path, default=Path("data/tokens"))
    ap.add_argument("--out", type=Path, default=Path("results/transplant"))
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--train-mb", type=int, default=0)
    ap.add_argument("--eval-batch", type=int, default=4)
    ap.add_argument("--eval-mb", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=40)
    args = ap.parse_args()

    todo = (
        TRANSPLANTS if args.all else [TRANSPLANTS_BY_NAME[n] for n in (args.exp or [])]
    )
    if not todo:
        raise SystemExit("pass --exp NAME [NAME ...] or --all")

    torch.set_float32_matmul_precision("high")
    print(f"loading host {HOST_MODEL} ...", flush=True)
    host = transplant.load_host(HOST_MODEL)
    print(f"host loaded: d_model={host[3]} vocab={host[1].shape[0]}", flush=True)

    tspec = TransplantTrainSpec()
    for spec in todo:
        run(spec, host, args, tspec)


if __name__ == "__main__":
    main()
