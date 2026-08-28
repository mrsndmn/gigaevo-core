"""Stage 0 of the tree-vocabulary program: distill the host embedding table.

Study 2 measured that the pretrained embedding table is worth ~0.62 BPB over a
retrained one (``orig-reinit``). This module asks the question everything
downstream depends on: **how many float parameters does that table actually
need?** Each method fits a structured approximation of the frozen table
offline -- no LM training anywhere -- then the reconstruction is plugged into
the frozen body (tied input embedding + output head, exactly like the native
table) and scored with study 2's bits-per-byte evaluator.

Methods, in increasing structure:

- ``svd``       global low-rank. The linear-algebra floor: any structured
                method that cannot beat it at matched parameters has no reason
                to exist.
- ``vq``        flat k-means codebook; every token snaps to one of K centroids.
- ``rvq``       residual VQ: the sum of M codebook entries. Additive
                composition without routing.
- ``tree``      hierarchical k-means, depth D branching B -- a decision tree
                over embedding space. A token's reconstruction is the sum of
                node vectors on its root-to-leaf path.
- ``tree_res``  the tree plus a shared low-rank residual basis with per-token
                coefficients: the tree carries the neighborhood, the residual
                carries token identity.

``weight="freq"`` variants minimize token-frequency-weighted MSE, with counts
taken from the cached qwen train tokens. The unweighted objective spends most
of its capacity on the Zipf tail the evaluation never sees; the weighted fit
is the honest version of "preserve the function, not the matrix".

Accounting is in float32 parameters. Discrete code indices are per-token
*storage* (a few bits each), not floats; they are reported separately and
never folded into the compression ratio.

Run::

    python -m experiments.vocab_factorization.distill_embed --all
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from experiments.vocab_factorization import perturb, transplant
from experiments.vocab_factorization.build_arms import get_arm
from experiments.vocab_factorization.configs import (
    DISTILLS,
    DISTILLS_BY_NAME,
    HOST_ARM,
    HOST_MODEL,
    DistillSpec,
    TransplantTrainSpec,
)
from experiments.vocab_factorization.prepare import build_text, encode_text, get_tokens
from experiments.vocab_factorization.train import (
    EVAL_SUITES,
    _heads_arrays,
    eval_bpb,
)
from experiments.vocab_factorization.train_transplant import _heads_for

SEQ_LEN = 1024  # matches study 2, so the numbers are directly comparable


# ---------------------------------------------------------------------------
# Fitting primitives (all fp32, on GPU, deterministic under the given seed)
# ---------------------------------------------------------------------------


def _kmeans(
    X: torch.Tensor,
    K: int,
    w: torch.Tensor | None,
    iters: int,
    seed: int,
    chunk: int = 32768,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(Weighted) k-means. Returns (centroids (K, d), assignment (N,))."""
    N = X.shape[0]
    K = min(K, N)
    g = torch.Generator(device=X.device).manual_seed(seed)
    if w is None:
        w = torch.ones(N, device=X.device)
    init = torch.multinomial(w, K, replacement=False, generator=g)
    C = X[init].clone()
    assign = torch.zeros(N, dtype=torch.long, device=X.device)
    for _ in range(iters):
        cn = (C * C).sum(1)
        for i in range(0, N, chunk):
            xb = X[i : i + chunk]
            d = cn[None, :] - 2.0 * (xb @ C.T)  # ||x||^2 is constant in argmin
            assign[i : i + chunk] = d.argmin(1)
        sums = torch.zeros_like(C)
        cnt = torch.zeros(K, device=X.device)
        sums.index_add_(0, assign, X * w[:, None])
        cnt.index_add_(0, assign, w)
        empty = cnt == 0
        C = sums / cnt.clamp(min=1e-12)[:, None]
        if empty.any():
            C[empty] = X[
                torch.randint(0, N, (int(empty.sum()),), generator=g, device=X.device)
            ]
    return C, assign


def _weighted_svd(
    W: torch.Tensor, w: torch.Tensor | None, max_rank: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-``max_rank`` factors (A, B) with W ~ A @ B under row-weighted MSE.

    Row weighting is exact for low-rank fitting: minimizing
    sum_i w_i ||W_i - (AB)_i||^2 equals unweighted SVD of diag(sqrt(w)) W with
    the row factor unscaled afterwards, because A is free per-row.
    """
    if w is None:
        scaled, unscale = W, None
    else:
        s = (w / w.mean()).clamp(min=1e-12).sqrt()
        scaled, unscale = W * s[:, None], s
    U, S, Vh = torch.linalg.svd(scaled, full_matrices=False)
    A = U[:, :max_rank] * S[:max_rank][None, :]
    if unscale is not None:
        A = A / unscale[:, None]
    return A, Vh[:max_rank]


def _fit_tree(
    W: torch.Tensor, B: int, D: int, w: torch.Tensor | None, iters: int, seed: int
) -> tuple[torch.Tensor, int, int]:
    """Hierarchical k-means. Returns (reconstruction, n_nodes, n_leaves).

    Each node stores the (weighted) mean of its cluster's residuals, so the
    sum of node vectors along a root-to-leaf path is exactly the leaf-cluster
    mean -- the decision-tree composition the program is built around.
    """
    recon = torch.zeros_like(W)
    groups = [torch.arange(W.shape[0], device=W.device)]
    n_nodes = 0
    for level in range(D):
        nxt: list[torch.Tensor] = []
        for gi, idx in enumerate(groups):
            R = W[idx] - recon[idx]
            wg = None if w is None else w[idx]
            C, a = _kmeans(R, B, wg, iters, seed + 1000 * level + gi)
            n_nodes += C.shape[0]
            recon[idx] += C[a]
            for k in range(C.shape[0]):
                sub = idx[a == k]
                if len(sub):
                    nxt.append(sub)
        groups = nxt
    return recon, n_nodes, len(groups)


def _fit_rvq(
    W: torch.Tensor, K: int, levels: int, w: torch.Tensor | None, iters: int, seed: int
) -> list[torch.Tensor]:
    """Residual VQ; returns per-level reconstruction increments (V, d)."""
    out: list[torch.Tensor] = []
    R = W.clone()
    for m in range(levels):
        C, a = _kmeans(R, K, w, iters, seed + m)
        inc = C[a]
        out.append(inc)
        R -= inc
    return out


# ---------------------------------------------------------------------------
# Composing a spec into (reconstruction, accounting), with shared-fit caching
# ---------------------------------------------------------------------------


class Fitter:
    """Fits are shared across specs: one SVD serves every rank, one rvq run
    serves every level count, one tree serves both tree arms. ``cache`` holds
    CPU tensors; compositions land back on the GPU per spec."""

    def __init__(
        self,
        W: torch.Tensor,
        weights: dict[str, torch.Tensor | None],
        iters: int,
        seed: int,
    ):
        self.W = W
        self.V, self.d = W.shape
        self.weights = weights
        self.iters = iters
        self.seed = seed
        self.cache: dict[str, object] = {}

    def _svd(self, weight: str, max_rank: int = 512):
        key = f"svd:{weight}"
        if key not in self.cache:
            t = time.time()
            A, Bm = _weighted_svd(self.W, self.weights[weight], max_rank)
            self.cache[key] = (A.cpu(), Bm.cpu())
            print(f"  fit {key} (rank<={max_rank}) {time.time() - t:.1f}s", flush=True)
        return self.cache[key]

    def _tree(self, B: int, D: int, weight: str):
        key = f"tree:{B}x{D}:{weight}"
        if key not in self.cache:
            t = time.time()
            recon, n_nodes, n_leaves = _fit_tree(
                self.W, B, D, self.weights[weight], self.iters, self.seed
            )
            self.cache[key] = (recon.cpu(), n_nodes, n_leaves)
            print(
                f"  fit {key}: nodes={n_nodes} leaves={n_leaves} "
                f"{time.time() - t:.1f}s",
                flush=True,
            )
        return self.cache[key]

    def _tree_residual(self, B: int, D: int, weight: str, max_rank: int = 448):
        key = f"treeres:{B}x{D}:{weight}"
        if key not in self.cache:
            recon, _, _ = self._tree(B, D, weight)
            t = time.time()
            R = self.W - recon.to(self.W.device)
            A, Bm = _weighted_svd(R, self.weights[weight], max_rank)
            self.cache[key] = (A.cpu(), Bm.cpu())
            print(f"  fit {key} (rank<={max_rank}) {time.time() - t:.1f}s", flush=True)
        return self.cache[key]

    def _rvq(self, K: int, levels: int, weight: str):
        key = f"rvq:{K}:{weight}"
        have = self.cache.get(key, [])
        if len(have) < levels:
            t = time.time()
            # Extend from what exists so rvq2/rvq4/rvq8 share a prefix.
            base = sum(
                (inc.to(self.W.device) for inc in have), torch.zeros_like(self.W)
            )
            more = _fit_rvq(
                self.W - base,
                K,
                levels - len(have),
                self.weights[weight],
                self.iters,
                self.seed + 7000 + len(have),
            )
            self.cache[key] = list(have) + [m.cpu() for m in more]
            print(
                f"  fit {key} levels {len(have)}->{levels} {time.time() - t:.1f}s",
                flush=True,
            )
        return self.cache[key][:levels]

    def compose(self, spec: DistillSpec) -> tuple[torch.Tensor, dict]:
        V, d = self.V, self.d
        native_floats = V * d
        if spec.method == "svd":
            A, Bm = self._svd(spec.weight)
            r = spec.rank
            W_hat = A[:, :r].to(self.W.device) @ Bm[:r].to(self.W.device)
            floats, bits = r * (V + d), 0
        elif spec.method == "vq":
            incs = self._rvq(spec.codes, 1, spec.weight)
            W_hat = incs[0].to(self.W.device).clone()
            floats = spec.codes * d
            bits = V * math.ceil(math.log2(spec.codes))
        elif spec.method == "rvq":
            incs = self._rvq(spec.codes, spec.levels, spec.weight)
            W_hat = sum((i.to(self.W.device) for i in incs), torch.zeros_like(self.W))
            floats = spec.levels * spec.codes * d
            bits = V * spec.levels * math.ceil(math.log2(spec.codes))
        elif spec.method == "tree":
            recon, n_nodes, n_leaves = self._tree(spec.codes, spec.depth, spec.weight)
            W_hat = recon.to(self.W.device).clone()
            floats = n_nodes * d
            bits = V * math.ceil(math.log2(max(2, n_leaves)))
        elif spec.method == "tree_res":
            recon, n_nodes, n_leaves = self._tree(spec.codes, spec.depth, spec.weight)
            A, Bm = self._tree_residual(spec.codes, spec.depth, spec.weight)
            r = spec.rank
            W_hat = recon.to(self.W.device) + A[:, :r].to(self.W.device) @ Bm[:r].to(
                self.W.device
            )
            floats = n_nodes * d + r * (V + d)
            bits = V * math.ceil(math.log2(max(2, n_leaves)))
        else:
            raise ValueError(spec.method)
        if spec.renorm:
            target = self.W.norm(dim=1, keepdim=True)
            W_hat = W_hat / W_hat.norm(dim=1, keepdim=True).clamp(min=1e-6) * target
            floats += V  # the stored per-row norms
        acct = {
            "float_params": floats,
            "index_bits": bits,
            "compression": native_floats / floats,
        }
        return W_hat, acct


# ---------------------------------------------------------------------------
# Reconstruction quality + evaluation
# ---------------------------------------------------------------------------


def recon_stats(
    W: torch.Tensor,
    W_hat: torch.Tensor,
    freq_w: torch.Tensor | None,
    active: torch.Tensor,
) -> dict:
    """Relative squared error (fraction of table variance unexplained), overall
    / frequency-weighted / restricted to token ids the eval text actually uses."""
    err = (W - W_hat).pow(2).sum(1)
    ref = W.pow(2).sum(1)

    def rel(mask_w: torch.Tensor | None) -> float:
        if mask_w is None:
            return float(err.sum() / ref.sum())
        return float((err * mask_w).sum() / (ref * mask_w).sum())

    cos = torch.nn.functional.cosine_similarity(W, W_hat, dim=1)
    return {
        "rel_err": rel(None),
        "rel_err_freq": rel(freq_w) if freq_w is not None else None,
        "rel_err_active": float(err[active].sum() / ref[active].sum()),
        "mean_cosine": float(cos.mean()),
    }


def evaluate_table(
    body,
    W_native: torch.Tensor,
    W_hat: torch.Tensor,
    apply: str,
    heads: dict[str, int],
    d_model: int,
    eval_sets: dict,
    suites: tuple[str, ...],
    device,
    eval_batch: int,
) -> dict:
    model = transplant.TransplantLM(
        body, heads, d_model, tie_embeddings=(apply == "both")
    ).to(device)
    with torch.no_grad():
        if apply == "both":
            model.emb["stem"].weight.copy_(W_hat)
        else:
            inp = W_hat if apply == "input" else W_native
            out = W_hat if apply == "output" else W_native
            model.emb["stem"].weight.copy_(inp)
            model.out["stem"].weight.copy_(out)
    res = {}
    for suite in suites:
        arrays, nb = eval_sets[suite]
        res[suite] = eval_bpb(model, arrays, nb, SEQ_LEN, device, eval_batch)
    del model
    # Deliberately no empty_cache(): on a shared GPU, memory returned to the
    # driver is snapped up by neighbor processes and never comes back.
    return res


# ---------------------------------------------------------------------------


def _reserve(gb: float) -> None:
    """Grab the run's peak memory once, up front.

    This node's GPU is shared and neighbor processes expand into any memory
    the moment it frees; per-allocation the race is unwinnable (two sweeps
    died to it). A large allocation that is immediately deleted stays in the
    caching allocator's pool, so the whole run then serves every allocation
    from memory it already owns. Pairs with never calling ``empty_cache``.
    """
    if gb <= 0:
        return
    buf = torch.empty(int(gb * (1 << 30)), dtype=torch.uint8, device="cuda")
    del buf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    ap.add_argument("--arms-dir", type=Path, default=Path("data/arms"))
    ap.add_argument("--tokens-dir", type=Path, default=Path("data/tokens"))
    ap.add_argument("--out", type=Path, default=Path("results/distill"))
    ap.add_argument("--eval-batch", type=int, default=4)
    ap.add_argument("--eval-mb", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--kmeans-iters", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reserve-gb", type=float, default=24.0)
    args = ap.parse_args()

    todo = DISTILLS if args.all else [DISTILLS_BY_NAME[n] for n in (args.exp or [])]
    if not todo:
        raise SystemExit("pass --exp NAME [NAME ...] or --all")

    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    _reserve(args.reserve_gb)
    # Halve the transient fp32 logits chunk (4096 x 151936 x 4B = 2.3 GB is
    # the single biggest allocation in the eval); 2048 rows keeps the peak
    # comfortably inside the reserved pool.
    transplant._CE_CHUNK = 2048
    print(f"loading host {HOST_MODEL} ...", flush=True)
    body, host_emb, _host_tok, d_model = transplant.load_host(HOST_MODEL)
    W = host_emb.float().to(device)
    V = W.shape[0]
    arm = get_arm(HOST_ARM, args.arms_dir, args.corpus)
    heads = _heads_for(arm, V)
    print(f"host loaded: d_model={d_model} table={tuple(W.shape)}", flush=True)

    # Token frequencies over the same 230 MB the transplants trained on
    # (cached), for the weighted fits and the weighted error metric.
    enc, _ = get_tokens(
        HOST_ARM,
        "train",
        TransplantTrainSpec().train_bytes,
        args.corpus,
        args.arms_dir,
        args.tokens_dir,
        workers=args.workers,
    )
    counts = np.bincount(enc.stem.astype(np.int64), minlength=V)
    freq_w = torch.from_numpy(counts + 1.0).float().to(device)
    print(f"freq weights: {int((counts > 0).sum())}/{V} ids seen in train", flush=True)

    valid_text = build_text(args.corpus, "valid", None)
    if args.eval_mb:
        cut = int(args.eval_mb * 1024 * 1024)
        valid_text = valid_text[:cut]
        valid_text = valid_text[: valid_text.rfind("\n") + 1]

    need = set(["clean"])
    for spec in todo:
        if spec.full_suites:
            need.update(EVAL_SUITES)
    eval_sets = {}
    for suite in EVAL_SUITES:
        if suite not in need:
            continue
        text = perturb.apply_suite(suite, valid_text, seed=0)
        e = encode_text(HOST_ARM, args.arms_dir, text, workers=args.workers)
        eval_sets[suite] = (_heads_arrays(e, heads), len(text.encode("utf-8")))
    active = torch.from_numpy(np.unique(eval_sets["clean"][0]["stem"])).to(device)
    suites_all = tuple(s for s in EVAL_SUITES if s in eval_sets)

    args.out.mkdir(parents=True, exist_ok=True)

    # Native reference, evaluated in this exact harness (sanity: must land on
    # study 2's orig-frozen 0.7125 when run on the full valid split).
    t = time.time()
    native = evaluate_table(
        body,
        W,
        W,
        "both",
        heads,
        d_model,
        eval_sets,
        suites_all,
        device,
        args.eval_batch,
    )
    ref = {
        "experiment": "native",
        "method": "none",
        "host": HOST_MODEL,
        "params": {"float_params": V * d_model, "index_bits": 0, "compression": 1.0},
        "eval_seconds": round(time.time() - t, 1),
    }
    for s in suites_all:
        ref[f"eval.{s}"] = native[s]
        ref[f"delta.{s}"] = native[s]["bpb"] - native["clean"]["bpb"]
    # A clean-only follow-up run must not clobber a full-suite native record:
    # merge new keys over whatever is already there.
    native_path = args.out / "native.json"
    if native_path.exists():
        ref = {**json.loads(native_path.read_text()), **ref}
    native_path.write_text(json.dumps(ref, indent=2))
    print(f"native clean bpb={native['clean']['bpb']:.4f}", flush=True)

    fitter = Fitter(W, {"none": None, "freq": freq_w}, args.kmeans_iters, args.seed)
    for spec in todo:
        t = time.time()
        W_hat, acct = fitter.compose(spec)
        stats = recon_stats(W, W_hat, freq_w, active)
        suites = suites_all if spec.full_suites else ("clean",)
        res = evaluate_table(
            body,
            W,
            W_hat,
            spec.apply,
            heads,
            d_model,
            eval_sets,
            suites,
            device,
            args.eval_batch,
        )
        del W_hat
        out = {
            "experiment": spec.name,
            "method": spec.method,
            "host": HOST_MODEL,
            "weight": spec.weight,
            "apply": spec.apply,
            "params": acct,
            "recon": stats,
            "note": spec.note,
            "eval_seconds": round(time.time() - t, 1),
        }
        for s in suites:
            out[f"eval.{s}"] = res[s]
            out[f"delta.{s}"] = res[s]["bpb"] - res["clean"]["bpb"]
        out["dnative.clean"] = res["clean"]["bpb"] - native["clean"]["bpb"]
        (args.out / f"{spec.name}.json").write_text(json.dumps(out, indent=2))
        print(
            f"[{spec.name}] x{acct['compression']:.1f} "
            f"rel_err={stats['rel_err']:.4f} "
            f"clean bpb={res['clean']['bpb']:.4f} "
            f"(+{out['dnative.clean']:.4f}) {time.time() - t:.0f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
