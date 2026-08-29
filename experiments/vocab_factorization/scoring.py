"""One routing matrix -> one fitness number, at a chosen ladder rung.

Shared by ``calibrate`` (which measures whether the cheap rungs track the
expensive one) and by ``problems/vocab_routing/validate.py`` (which spends
them). Keeping one implementation is the point: a calibration that measured a
different L1 from the one evolution then optimizes would certify the wrong
thing, and the failure would be invisible because both numbers would look
reasonable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from experiments.vocab_factorization import active_vocab, perturb, transplant
from experiments.vocab_factorization.configs import (
    HOST_MODEL,
    TransplantTrainSpec,
    TreeSpec,
)
from experiments.vocab_factorization.prepare import build_text, encode_text
from experiments.vocab_factorization.train import WindowSampler, _heads_arrays, eval_bpb
from experiments.vocab_factorization.train_transplant import _heads_for

SEQ_LEN = 1024
RUNGS = ("L0", "L1", "L2", "L3")
_L2_STEPS = 60
_L3_STEPS = 0  # 0 = the full schedule


def score_routing(
    matrix: np.ndarray,
    arm,
    host_tok,
    enc,
    rung: str = "L2",
    data: Path = Path("data"),
    n_trees: int = 16,
    leaves: int = 124,
    seed: int = 0,
    eval_mb: float = 0.5,
) -> dict:
    """Fit a partition from ``matrix`` and score it at ``rung``."""
    from experiments.vocab_factorization import tree_embed

    if rung not in RUNGS:
        raise ValueError(f"unknown rung {rung!r}, have {RUNGS}")

    vocab = arm.heads["stem"]
    counts = np.bincount(enc.stem.astype(np.int64), minlength=vocab).astype(np.float64)

    if rung == "L0":
        # No body, no GPU: the partition is scored against the mean-of-pieces
        # geometry alone. Stage 0 says this does not predict BPB; the ladder
        # exists to find out how badly.
        from experiments.vocab_factorization.calibrate import l0_explained_variance

        targets = _mean_of_pieces_cpu(arm, host_tok, vocab)
        routing = tree_embed.fit_partition(
            matrix,
            targets,
            TreeSpec(n_trees=n_trees, leaves_per_tree=leaves, seed=seed),
        )
        ev = l0_explained_variance(targets, routing.table, counts + 1.0)
        flat = active_vocab.active_floats(int((counts > 0).sum()), targets.shape[1])
        return {
            "rung": "L0",
            # Mapped into BPB's direction so one fitness expression serves every
            # rung; the units are not BPB and the calibration is what says
            # whether the ordering is the same.
            "bpb": float(1.0 - ev),
            "explained_variance": float(ev),
            "compression": flat / max(1, routing.n_leaves * targets.shape[1]),
        }

    device = torch.device("cuda")
    tspec = TransplantTrainSpec()
    body, host_emb, _tok, d_model = transplant.load_host(HOST_MODEL)
    heads = _heads_for(arm, host_emb.shape[0])
    model = transplant.TransplantLM(body, heads, d_model, tie_embeddings=True).to(
        device
    )
    transplant.mean_of_pieces_init(model, arm, host_tok, host_emb, verbose=False)
    targets = model.emb["stem"].weight.detach().float().cpu().numpy()

    routing = tree_embed.fit_partition(
        matrix, targets, TreeSpec(n_trees=n_trees, leaves_per_tree=leaves, seed=seed)
    )
    tree = tree_embed.TreeEmbedding(routing, d_model).to(device)
    model.emb["stem"] = tree
    model.out["stem"] = tree

    ids = active_vocab.load_or_build(
        arm.name if hasattr(arm, "name") else "bpe8k",
        vocab,
        data / "corpus",
        data / "arms",
        data / "tokens",
        tspec.train_bytes,
    )
    model.set_active_mask(active_vocab.mask_from_ids(ids, heads["stem"], device))
    flat = active_vocab.active_floats(int(ids.size), d_model)
    acct = tree.accounting(flat)

    valid_text = build_text(data / "corpus", "valid", None)
    if rung != "L3":
        cut = int(eval_mb * 1024 * 1024)
        valid_text = valid_text[:cut]
        valid_text = valid_text[: valid_text.rfind("\n") + 1]
    e = encode_text(
        arm.name if hasattr(arm, "name") else "bpe8k",
        data / "arms",
        perturb.apply_suite("clean", valid_text, 0),
    )
    arrays = _heads_arrays(e, heads)
    nbytes = len(valid_text.encode("utf-8"))

    steps = {"L1": 0, "L2": _L2_STEPS, "L3": _L3_STEPS}[rung]
    if steps or rung == "L3":
        n_steps = steps or int(len(enc) / tspec.batch_tokens)
        _train(model, body, enc, heads, tspec, n_steps, device)

    bpb = eval_bpb(model, arrays, nbytes, SEQ_LEN, device, 4)["bpb"]
    del model
    return {"rung": rung, "bpb": float(bpb), "compression": acct["compression"]}


def _train(model, body, enc, heads, tspec, n_steps: int, device, micro: int = 16):
    sampler = WindowSampler(_heads_arrays(enc, heads), SEQ_LEN, tspec.seed, device)
    params = model.trainable_parameters()
    opt = torch.optim.AdamW(params, lr=tspec.lr, betas=(0.9, 0.95), weight_decay=0.0)
    model.train()
    body.eval()
    for _ in range(n_steps):
        opt.zero_grad(set_to_none=True)
        for _ in range(2):
            inp, tgt = sampler.batch(micro)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                losses = model(inp, tgt)
            (sum(losses.values()) / (micro * SEQ_LEN * 2)).backward()
        torch.nn.utils.clip_grad_norm_(params, tspec.grad_clip)
        opt.step()


def _mean_of_pieces_cpu(arm, host_tok, vocab: int) -> np.ndarray:
    """Mean-of-pieces targets without loading the 1.7B body onto a GPU.

    L0 has to be seconds on a CPU or it is not a cheap rung, and the host's
    embedding table alone is enough to build the targets it scores.
    """
    from transformers import AutoModelForCausalLM

    from experiments.vocab_factorization.tokenization import piece_string

    lm = AutoModelForCausalLM.from_pretrained(HOST_MODEL, dtype=torch.float32)
    table = lm.get_input_embeddings().weight.detach()
    del lm
    out = np.zeros((vocab, table.shape[1]), dtype=np.float32)
    for vid in range(vocab):
        s = piece_string(arm, vid)
        pieces = host_tok(s, add_special_tokens=False)["input_ids"] if s else []
        if pieces:
            out[vid] = table[torch.tensor(pieces)].mean(0).numpy()
    return out
