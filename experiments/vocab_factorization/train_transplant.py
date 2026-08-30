"""Run the vocabulary-transplant grid on a frozen pretrained body.

Reuses study 1's corpus, tokenizer arms, perturbation suites, and — crucially —
its bits-per-byte evaluator, so the two sets of numbers are directly
comparable in kind (not in value: different body, different corpus budget).

Every arm is evaluated **twice**: once at step 0, before any training, and once
after. The step-0 row under ``mean_of_pieces`` init is the literal
"modify the vocabulary on the fly, train nothing" measurement.

Trained arms write their embedding and heads to ``--ckpt-dir``. Nothing else is
saved because nothing else changes: the body is frozen and reloads from HF. This
is what lets a downstream evaluation ride along on a run scheduled for some other
reason instead of paying for its own.

Every arm is optimized and scored under the **active-vocabulary mask** (see
``active_vocab``), so the softmax normalizer is the same set of competitors for
all of them. ``--no-mask`` reproduces the pre-mask numbers and nothing else:
masked and unmasked BPB are not comparable. A progressive-shrink arm
(``--all-shrink``) is the exception, and only because it goes further: its
vocabulary has already had the inactive rows *deleted*, so there is nothing left
for a mask to remove.

Run::

    python -m experiments.vocab_factorization.train_transplant --all
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from experiments.vocab_factorization import active_vocab, perturb, transplant
from experiments.vocab_factorization.build_arms import get_arm
from experiments.vocab_factorization.configs import (
    HOST_MODEL,
    MASKED_NATIVE_BPB,
    SHRINK_CURRICULUM,
    SHRINK_GATE_BPB,
    SHRINKS,
    SHRINKS_BY_NAME,
    TRANSPLANTS,
    TRANSPLANTS_BY_NAME,
    TREE_TRANSPLANTS,
    TREE_TRANSPLANTS_BY_NAME,
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


def _save_checkpoint(model, spec, tspec, heads, steps, path: Path) -> str:
    """Persist the trained parameters. A frozen body reloads from HF instead.

    Round 1 of this study saved nothing, so five arms' worth of trained tables
    were unrecoverable and any downstream question about them required a full
    re-train first. ``emb`` and ``out`` *are* the transplant; ``torch.save``
    keeps ``out.stem`` sharing storage with ``emb.stem`` while they are tied,
    so a tied arm costs one table on disk, not two.

    The body is stored **only** when the arm trained it. For every frozen arm
    it is bit-identical to the HF weights and storing it would cost 3.4 GB per
    arm to duplicate a file that already exists; for the unfrozen arm it is the
    thing that changed, and omitting it would produce a checkpoint that reloads
    silently into the wrong model — the pristine body paired with an embedding
    trained against a different one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "experiment": spec.name,
        "arm": spec.arm,
        "host": HOST_MODEL,
        "init": spec.init,
        "heads": heads,
        "steps": steps,
        "seed": tspec.seed,
        "freeze_body": spec.freeze_body,
        # A shrink arm's heads alone do not say which vocabulary they index, and
        # the codec is derived rather than stored (it is a deterministic function
        # of the arm, the active set, the curriculum prefix and the train
        # stream). Recording the prefix length is what makes the checkpoint
        # self-describing enough to rebuild it.
        "stage": getattr(spec, "stage", 0),
        "emb": model.emb.state_dict(),
        "out": model.out.state_dict(),
    }
    if not spec.freeze_body:
        blob["body"] = model.body.state_dict()
    torch.save(blob, path)
    return str(path)


def _reserve(gb: float) -> None:
    """Grab the run's peak memory once, up front. See ``distill_embed._reserve``."""
    if gb <= 0:
        return
    buf = torch.empty(int(gb * (1 << 30)), dtype=torch.uint8, device="cuda")
    del buf


@torch.no_grad()
def _tree_targets(model, spec, sampler, tspec, micro: int, rounds: int = 8):
    """What the partition is fit to.

    Default: the mean-of-pieces vectors already sitting in the embedding after
    init. They define the partition only — stage 0 measured that *fitted values*
    from this kind of target cost at least +1.37 BPB, so the leaves are relearned
    through the body regardless.

    ``functional_boost``: the accumulated loss gradient w.r.t. each embedding
    row, collected through the frozen body. This is the correction stage 0's
    failure demands — grouping tokens that the *objective* wants moved the same
    way, rather than tokens whose mean-of-pieces vectors happen to be close.
    It is the single-pass form of the idea: one gradient field, fit once, rather
    than re-collected between boosting rounds.
    """
    if not getattr(spec, "functional_boost", False):
        return model.emb["stem"].weight.detach().float().cpu().numpy()

    with torch.enable_grad():
        model.emb["stem"].weight.grad = None
        for _ in range(rounds):
            inputs, targets = sampler.batch(micro)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                losses = model(inputs, targets)
            (sum(losses.values()) / (micro * tspec.seq_len * rounds)).backward()
    grad = model.emb["stem"].weight.grad
    model.emb["stem"].weight.grad = None
    return grad.detach().float().cpu().numpy()


def _install_tree(model, spec, features, targets, d_model: int, n_active: int) -> dict:
    """Swap the flat stem table for a feature-routed one. The other hook.

    Both sides are replaced with the *same* module, so the tie is structural:
    ``TreeEmbedding.weight`` composes the table from the shared leaves, which is
    the "output embedding composed from the same tied structure" the plan
    requires and the side stage 0 found to be the sensitive one.
    """
    from experiments.vocab_factorization import tree_embed
    from experiments.vocab_factorization.active_vocab import active_floats

    # An HF host's embedding table is padded past ``len(tokenizer)`` — 151,936
    # rows for 151,669 ids — and features are extracted per *tokenizer* id, so
    # the two disagree by exactly the padding. Every table row still needs a
    # route or the routed table would have fewer rows than the softmax it feeds.
    # Zero rows put the padding in one leaf together, which is right: they are
    # unreachable by the tokenizer and masked out of the softmax anyway. On a
    # corpus-trained arm the gap is zero and this is a no-op, which is why only
    # the host bridge arm ever hit it.
    rows = targets.shape[0]
    if features.shape[0] != rows:
        if features.shape[0] > rows:
            raise ValueError(
                f"{features.shape[0]} feature rows for a {rows}-row table — "
                "features cannot describe tokens the table does not have"
            )
        pad = np.zeros((rows - features.shape[0], features.shape[1]), features.dtype)
        features = np.concatenate([features, pad], axis=0)

    flat_floats = active_floats(n_active, d_model)
    leaves = spec.tree.leaves_per_tree or tree_embed.leaves_for_budget(
        flat_floats, spec.budget_ratio, d_model, spec.tree.n_trees, spec.tree.film
    )
    tspec = replace(spec.tree, leaves_per_tree=leaves)
    routing = tree_embed.fit_partition(features, targets, tspec)
    tree = tree_embed.TreeEmbedding(routing, d_model, film=tspec.film).to(
        model.emb["stem"].weight.device
    )
    model.emb["stem"] = tree
    model.out["stem"] = tree
    return tree.accounting(flat_floats)


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

    # The re-indexing hook (progressive-shrink), and the third distinct thing
    # this loop can do to a vocabulary. The other two leave the id space alone:
    # the active-vocab mask restricts the softmax over a table that keeps all
    # its rows, and the tree hook swaps the module behind a fixed row count.
    # A shrink step *re-indexes* — it deletes rows and rewrites the stream — so
    # it has to run before the model is built, and it composes with the mask by
    # subsuming it: every row of a shrunk table is reachable from an active host
    # id by construction, so there is nothing left to mask and the mask block
    # below is skipped rather than applied on top. Rule 0 makes that exact
    # rather than approximate — it *is* the mask, expressed as deletion.
    plan = shrink_acct = host_counts = None
    if getattr(spec, "stage", 0):
        from experiments.vocab_factorization import shrink

        active_ids = active_vocab.load_or_build(
            spec.arm,
            arm.heads["stem"],
            args.corpus,
            args.arms_dir,
            args.tokens_dir,
            tspec.train_bytes,
            workers=args.workers,
        )
        # Overriding the last rule's threshold rather than extending the chain
        # keeps every arm a nested prefix of one curriculum — which is what
        # makes the result a curve rather than a scatter of unrelated
        # vocabularies. A longer chain would apply a 200 threshold to a
        # vocabulary already thresholded at 1000, which is a different thing.
        rules = list(SHRINK_CURRICULUM[: spec.stage])
        if getattr(spec, "override_min_count", 0) and rules:
            rules[-1] = replace(rules[-1], min_count=spec.override_min_count)
        plan = shrink.build(
            shrink.host_vocabulary(arm, heads["stem"]),
            active_ids,
            enc.stem,
            rules,
        )
        host_counts = np.bincount(enc.stem.astype(np.int64), minlength=heads["stem"])
        heads = plan.heads()
        train_arrays = plan.apply_stream(enc.stem)
        shrink_acct = {**plan.accounting(d_model), **plan.summary()}
        print(
            f"[{spec.name}] shrink stage {spec.stage}: "
            f"{[s.rule for s in plan.steps]} -> {heads} "
            f"{shrink_acct['float_params'] / 1e6:.2f}M embedding floats "
            f"(x{shrink_acct['compression_vs_host']:.2f} vs the host table), "
            f"tokens {shrink_acct['token_inflation']:+.2%}",
            flush=True,
        )
    else:
        train_arrays = _heads_arrays(enc, heads)
    n_tokens = int(next(iter(train_arrays.values())).shape[0])

    model = transplant.TransplantLM(
        body, heads, d_model, tie_embeddings=True, freeze_body=spec.freeze_body
    ).to(device)
    if spec.init == "shrink":
        if plan is None:
            raise ValueError(
                f"{spec.name} asks for the shrink init but carries no stage — "
                "there is no codec to fold the host table down"
            )
        init_stats = transplant.shrink_init(
            model, plan.source_rows(host_counts), host_emb
        )
    elif spec.init == "native":
        init_stats = transplant.native_init(model, host_emb)
    elif spec.init == "random":
        init_stats = transplant.random_init(model, host_emb)
    elif spec.init == "mean_of_pieces":
        init_stats = transplant.mean_of_pieces_init(model, arm, host_tok, host_emb)
    else:
        raise ValueError(spec.init)

    seq = tspec.seq_len
    micro = spec.micro_batch

    # The embedding-swap hook. It runs after init because the mean-of-pieces
    # vectors *are* the default fit target, and before the optimizer is built
    # because it changes which parameters exist.
    tree_acct = None
    if getattr(spec, "budget_ratio", None):
        from experiments.vocab_factorization import token_features

        active_ids = active_vocab.load_or_build(
            spec.arm,
            arm.heads["stem"],
            args.corpus,
            args.arms_dir,
            args.tokens_dir,
            tspec.train_bytes,
            workers=args.workers,
        )
        feats = token_features.extract(
            arm, spec.features, host_tok=host_tok, stream=enc.stem
        )
        sampler = WindowSampler(train_arrays, seq, tspec.seed, device)
        targets = _tree_targets(model, spec, sampler, tspec, micro)
        del sampler
        tree_acct = _install_tree(
            model, spec, feats.matrix, targets, d_model, int(active_ids.size)
        )
        print(
            f"[{spec.name}] routed table: {tree_acct['n_leaves']} leaves over "
            f"{tree_acct['n_trees']} trees, {tree_acct['float_params'] / 1e6:.2f}M "
            f"floats vs {tree_acct['flat_floats'] / 1e6:.1f}M flat "
            f"(x{tree_acct['compression']:.2f}), "
            f"{tree_acct['index_bits'] // heads['stem']} route bits/token",
            flush=True,
        )

    counts = model.param_counts()
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
        # A shrunk arm scores the same *bytes* through a different token stream,
        # which is exactly why the metric is bits per byte: the denominator does
        # not move when re-piecing lengthens the numerator's stream.
        arrays = plan.apply_stream(e.stem) if plan else _heads_arrays(e, heads)
        eval_sets[suite] = (arrays, len(text.encode("utf-8")))

    # The active-vocabulary convention, applied identically to every arm. Built
    # from this arm's own streams, so an arm whose vocabulary was trained on the
    # corpus satisfies it very nearly trivially while the host's 152k table
    # sheds the half of itself this ru+en corpus never exercises. Skipping it is
    # possible (--no-mask) but only for reproducing the pre-mask numbers: an
    # unmasked arm's BPB is not comparable to a masked one's.
    active_ids = mask_info = None
    if not args.no_mask and plan is None:
        active_ids = active_vocab.load_or_build(
            spec.arm,
            arm.heads["stem"],
            args.corpus,
            args.arms_dir,
            args.tokens_dir,
            tspec.train_bytes,
            workers=args.workers,
        )
        for suite in EVAL_SUITES:
            active_vocab.assert_targets_active(
                eval_sets[suite][0], active_ids, f"{spec.name}/{suite}"
            )
        model.set_active_mask(
            active_vocab.mask_from_ids(active_ids, heads["stem"], device)
        )
        mask_info = {
            "n_active": int(active_ids.size),
            "table_size": heads["stem"],
            "active_flat_floats": active_vocab.active_floats(active_ids.size, d_model),
        }
        print(
            f"[{spec.name}] active vocab {active_ids.size}/{heads['stem']} "
            f"({active_ids.size / heads['stem']:.1%}), "
            f"flat table {mask_info['active_flat_floats'] / 1e6:.1f}M floats",
            flush=True,
        )

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

        sampler = WindowSampler(train_arrays, seq, tspec.seed, device)
        model.train()
        if spec.freeze_body:
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

    # Only trained arms are saved: an untrained ``native`` arm's state is the
    # host's own table, which reloads from HF for free.
    ckpt = None
    if steps and not args.no_ckpt:
        ckpt = _save_checkpoint(
            model, spec, tspec, heads, steps, args.ckpt_dir / f"{spec.name}.pt"
        )
        print(f"[{spec.name}] checkpoint -> {ckpt}", flush=True)

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
        "checkpoint": ckpt,
        "active_vocab": mask_info,
        "tree": tree_acct,
        "shrink": shrink_acct,
        "note": spec.note,
    }
    for suite in EVAL_SUITES:
        results[f"eval.{suite}"] = post[suite]
        results[f"step0.{suite}"] = pre[suite]
    clean = post["clean"]["bpb"]
    for suite in EVAL_SUITES:
        results[f"delta.{suite}"] = post[suite]["bpb"] - clean
    results["step0_clean_bpb"] = pre["clean"]["bpb"]
    if shrink_acct is not None:
        # The gate, evaluated in the artifact rather than left to whoever reads
        # it: continue the curriculum while healed BPB stays within +0.01 of the
        # masked native reference, stop at the vocabulary size where it does not.
        results["gate_margin"] = MASKED_NATIVE_BPB + SHRINK_GATE_BPB - clean
        results["gate_pass"] = bool(results["gate_margin"] >= 0)
        print(
            f"[{spec.name}] gate: clean {clean:.4f} vs "
            f"{MASKED_NATIVE_BPB} + {SHRINK_GATE_BPB} -> "
            f"{'PASS' if results['gate_pass'] else 'STOP'} "
            f"({results['gate_margin']:+.4f}) at V={heads['stem']}",
            flush=True,
        )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{spec.name}.json").write_text(json.dumps(results, indent=2))
    del model
    torch.cuda.empty_cache()
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--all-trees", action="store_true", help="the stage-1 grid")
    ap.add_argument(
        "--all-shrink", action="store_true", help="the progressive-shrink curve"
    )
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    ap.add_argument("--arms-dir", type=Path, default=Path("data/arms"))
    ap.add_argument("--tokens-dir", type=Path, default=Path("data/tokens"))
    ap.add_argument("--out", type=Path, default=Path("results/transplant"))
    ap.add_argument("--ckpt-dir", type=Path, default=Path("results/transplant/ckpt"))
    ap.add_argument("--no-ckpt", action="store_true")
    ap.add_argument(
        "--no-mask",
        action="store_true",
        help="evaluate over the full table (reproduces the pre-mask numbers)",
    )
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--train-mb", type=int, default=0)
    ap.add_argument("--eval-batch", type=int, default=4)
    ap.add_argument("--eval-mb", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--reserve-gb", type=float, default=0.0)
    args = ap.parse_args()

    # One loop drives all three grids: every arm carries the same execution
    # fields and differs only by which hook it keys on — ``budget_ratio`` for
    # the embedding swap, ``stage`` for the vocabulary re-indexing.
    known = {**TRANSPLANTS_BY_NAME, **TREE_TRANSPLANTS_BY_NAME, **SHRINKS_BY_NAME}
    if args.all:
        todo = list(TRANSPLANTS)
    elif args.all_trees:
        todo = list(TREE_TRANSPLANTS)
    elif args.all_shrink:
        todo = list(SHRINKS)
    else:
        todo = [known[n] for n in (args.exp or [])]
    if not todo:
        raise SystemExit("pass --exp NAME [NAME ...] or --all")

    torch.set_float32_matmul_precision("high")
    # Grab the run's peak memory once. Stage 0 lost two launches to neighbour
    # processes expanding into memory freed between allocations; on a dedicated
    # MLS worker this is a no-op, so it costs nothing to leave on. Pairs with
    # never calling empty_cache mid-run.
    _reserve(args.reserve_gb)
    print(f"loading host {HOST_MODEL} ...", flush=True)
    host = transplant.load_host(HOST_MODEL)
    print(f"host loaded: d_model={host[3]} vocab={host[1].shape[0]}", flush=True)

    tspec = TransplantTrainSpec()
    for spec in todo:
        run(spec, host, args, tspec)


if __name__ == "__main__":
    main()
