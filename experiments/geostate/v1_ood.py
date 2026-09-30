"""H4: the trained V1 adapters on poses outside the training range.

    python -m experiments.geostate.v1_ood --prepare [--out DIR] [--smoke]
    python -m experiments.geostate.v1_ood --eval --arm tokens --seed 1 [--out DIR] [--smoke]
    python -m experiments.geostate.v1_ood --report [--out DIR]

H1 compared the arms on poses drawn like the training ones (integer given
points in [-10, 10]); there the text table's coordinates are simply more
information. The remaining argument for invariant state tokens is that the
state blocks do not move when the figure does. This module re-evaluates the V1
adapters of the protocol H1 read (`V1_OOD.source_protocol`) — no training —
on the frozen V1 evaluation set mapped through each `PoseCondition` in
`configs.py`, and reads the H4 gate on the far conditions.

What moves under a far pose: in *both* arms the prompt's given coordinates
(`sequence.prompt_text`); in the table arm also every state row. The tokens
arm's state blocks are invariant (features match to float precision). H4 is
therefore about robustness: the gate reads the paired per-problem *drop* of
each arm from identity to a far condition, tokens drop minus table drop, not
the absolute gap (which starts ~19 pts in the table's favour).

Each condition keeps the original problem ids, so episodes pair across arms,
seeds and conditions by id. A problem a condition cannot build (a pose the
kernel finds degenerate, a target that drifts under rounding, an
orientation-dependent target under a reflection, the generator's newer
guards) or builds as a *different* construction (rounding breaks an exact
integer-grid coincidence and the reference changes length) is dropped from
that condition only and counted in `prepare.json`. The identity condition
must reproduce the saved V1 episodes; the verdict is withheld otherwise.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import fcntl
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np

from .cga import Similarity
from .client import read_json, write_json
from .configs import (
    ARM_TABLE,
    ARM_TOKENS,
    ARTIFACT_ROOT,
    H4_GATE,
    LENGTH_BUCKETS,
    OUT_DIR,
    TOL,
    V1,
    V1_OOD,
    PoseCondition,
    ToleranceSpec,
    V1Config,
    V1OODConfig,
)
from .generator import Problem, load_problems, save_problems
from .state import MoveError

# ---------------------------------------------------------------------------
# Poses and frozen inputs
# ---------------------------------------------------------------------------


def pose_similarity(cond: PoseCondition, rng: random.Random) -> Similarity:
    theta = rng.uniform(0.0, 2.0 * math.pi) if cond.rotate else 0.0
    phi = rng.uniform(0.0, 2.0 * math.pi)
    return Similarity(
        tx=cond.translation * math.cos(phi),
        ty=cond.translation * math.sin(phi),
        theta=theta,
        scale=cond.scale,
        reflect=cond.reflect,
    )


def _is_random(cond: PoseCondition) -> bool:
    return cond.rotate or cond.translation > 0


def _rng(cfg: V1OODConfig, cond: PoseCondition, problem_id: str) -> random.Random:
    # per problem and condition, so a pose never depends on file order
    digest = hashlib.sha256(f"{cfg.seed}/{cond.name}/{problem_id}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def pose_problem(
    problem: Problem,
    cond: PoseCondition,
    cfg: V1OODConfig = V1_OOD,
    tol: ToleranceSpec = TOL,
) -> Problem | None:
    """The problem under one pose of `cond`, keeping its id; None if unbuildable.

    The posed problem must be the same construction: the same reference length
    always, and the same trajectory under an orientation-preserving pose (a
    reflection legitimately swaps which point of an intersection pair a move
    names). Rounding a rotated integer grid can break an exact coincidence the
    original reference relied on — a 3-move problem then needs 20 — and such
    a copy would pair problems of different difficulty.
    """
    from .consistency import transformed_problem

    rng = _rng(cfg, cond, problem.id)
    identity = {label: label for label in problem.given}
    original = [m.to_dict() for m in problem.trajectory]
    for _ in range(cfg.max_attempts if _is_random(cond) else 1):
        g = pose_similarity(cond, rng)
        try:
            copy = transformed_problem(problem, g, identity, tol, 0)
        except (MoveError, ValueError):
            continue
        if copy.n_moves != problem.n_moves:
            continue
        if not cond.reflect and [m.to_dict() for m in copy.trajectory] != original:
            continue
        return replace(
            copy,
            id=problem.id,
            meta={**copy.meta, "condition": cond.name, "pose": g.to_dict()},
        )
    return None


def ood_hash(cfg: V1OODConfig = V1_OOD, v1: V1Config = V1) -> str:
    """Config plus every input on the path from a frozen problem to an episode.

    Whole modules, as `run_v1_jobs.protocol_hash_v1` does: an unrelated edit to
    `generator` or `rollout` also yields a new directory, never a silent
    resume under changed code. Report code is outside it.
    """
    from . import (
        cga,
        consistency,
        encoder,
        features,
        generator,
        macros,
        prompts,
        rollout,
        sequence,
        state,
        v1_model,
    )

    modules = (
        cga,
        state,
        macros,
        generator,
        prompts,
        features,
        sequence,
        encoder,
        v1_model,
        rollout,
    )
    material = json.dumps(
        {
            "config": asdict(cfg),
            "v1": asdict(v1),
            "tolerances": asdict(TOL),
            "buckets": LENGTH_BUCKETS,
            "code": {m.__name__: inspect.getsource(m) for m in modules},
            "poses": [
                inspect.getsource(f)
                for f in (
                    pose_similarity,
                    _rng,
                    pose_problem,
                    consistency.transformed_problem,
                    consistency._round2,
                )
            ],
        },
        sort_keys=True,
    )
    return hashlib.sha256(material.encode()).hexdigest()[:12]


def default_out(cfg: V1OODConfig = V1_OOD, smoke: bool = False) -> Path:
    if smoke:
        return OUT_DIR / f"{cfg.name}-smoke"
    return Path(ARTIFACT_ROOT) / f"{cfg.name}-{ood_hash(cfg)}"


def default_source_root() -> Path:
    return Path(ARTIFACT_ROOT) / "v1"


def source_dir(cfg: V1OODConfig, root: Path, arm: str, seed: int) -> Path:
    return Path(root) / f"{arm}-s{seed}-{cfg.source_protocol}"


def eval_set_path(cfg: V1OODConfig, root: Path) -> Path:
    return Path(root) / f"data-{cfg.data_key}" / "eval.jsonl"


def smoke_subset(problems: list[Problem], per_bucket: int) -> list[Problem]:
    taken: dict[str, int] = {}
    out = []
    for p in problems:
        if taken.get(p.bucket, 0) < per_bucket:
            taken[p.bucket] = taken.get(p.bucket, 0) + 1
            out.append(p)
    return out


def fingerprint(
    cfg: V1OODConfig, source_root: Path, smoke: bool = False
) -> dict[str, Any]:
    # Content, not paths: the launcher prepares under /mnt, the job reads the
    # same directory under /workspace.
    eval_bytes = eval_set_path(cfg, source_root).read_bytes()
    return {
        "ood": ood_hash(cfg),
        "smoke": smoke,
        "eval_set_sha256": hashlib.sha256(eval_bytes).hexdigest(),
    }


def prepare(
    cfg: V1OODConfig,
    out: Path,
    source_root: Path,
    smoke: bool = False,
    tol: ToleranceSpec = TOL,
) -> dict[str, Any]:
    """Freeze one problem file per condition (once; concurrent callers wait)."""
    manifest = out / "prepare.json"
    fp = fingerprint(cfg, source_root, smoke)

    def check(existing: dict[str, Any]) -> dict[str, Any]:
        if existing.get("fingerprint") != fp:
            raise RuntimeError(
                f"{out} was prepared for {existing.get('fingerprint')}, not {fp}; "
                "use another --out"
            )
        return existing

    if manifest.exists():
        return check(read_json(manifest))
    out.mkdir(parents=True, exist_ok=True)
    with (out / "prepare.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if manifest.exists():
            return check(read_json(manifest))
        problems = load_problems(eval_set_path(cfg, source_root))
        if smoke:
            problems = smoke_subset(problems, cfg.smoke_per_bucket)
        cond_dir = out / "conditions"
        cond_dir.mkdir(exist_ok=True)
        counts: dict[str, dict[str, Any]] = {}
        for cond in cfg.conditions:
            kept: list[Problem] = []
            rejected: list[str] = []
            for p in problems:
                posed = pose_problem(p, cond, cfg, tol)
                if posed is None:
                    rejected.append(p.id)
                else:
                    kept.append(posed)
            path = cond_dir / f"{cond.name}.jsonl"
            tmp = path.with_suffix(".jsonl.tmp")
            save_problems(kept, tmp)
            os.replace(tmp, path)
            counts[cond.name] = {
                "kept": len(kept),
                "rejected": len(rejected),
                "rejected_ids": rejected,
            }
            print(
                f"[{cfg.name}] {cond.name}: {len(kept)} kept, {len(rejected)} rejected",
                flush=True,
            )
        payload = {
            "fingerprint": fp,
            "n_source": len(problems),
            "conditions": counts,
            "config": asdict(cfg),
        }
        write_json(manifest, payload)  # written last: its presence means complete
    return read_json(manifest)  # the same shape on first and later calls


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def run_dir(out: Path, arm: str, seed: int) -> Path:
    return out / f"{arm}-s{seed}"


def evaluate_run(
    cfg: V1OODConfig,
    arm: str,
    seed: int,
    out: Path,
    source_root: Path,
    smoke: bool = False,
    device: str = "cuda",
    model=None,
    adapters: Path | None = None,
) -> dict[str, Any]:
    """Roll out one arm x seed's adapters on every condition (resumable)."""
    from .rollout import evaluate

    info = prepare(cfg, out, source_root, smoke)
    adapters = adapters or source_dir(cfg, source_root, arm, seed) / "adapters.pt"
    if model is None:
        from .train_v1 import seed_everything
        from .v1_model import V1Model

        # as the V1 run did; the adapters carry every trained parameter and
        # the <st> row is never read (its embedding is the encoder's output),
        # so this only keeps any stray randomness identical to that run
        seed_everything(seed)
        model = V1Model(V1, arm, device=device)
    if model.arm != arm:
        raise ValueError(f"model is for arm {model.arm!r}, asked for {arm!r}")
    model.load_adapters(adapters)
    run = run_dir(out, arm, seed)
    results: dict[str, Any] = {}
    for cond in cfg.conditions:
        problems = load_problems(out / "conditions" / f"{cond.name}.jsonl")
        print(f"[{arm} s{seed}] {cond.name}: {len(problems)} problems", flush=True)
        metrics = evaluate(model, problems, V1, run / cond.name)
        results[cond.name] = {"n": metrics["n"], "complete": metrics["complete"]}
    payload = {
        "arm": arm,
        "seed": seed,
        "adapters": str(adapters),
        "ood": info["fingerprint"]["ood"],
        "smoke": smoke,
        "conditions": results,
        "complete": all(r["complete"] for r in results.values()),
    }
    write_json(run / "run.json", payload)
    return payload


# ---------------------------------------------------------------------------
# Report and the H4 gate
# ---------------------------------------------------------------------------


def _read_episodes(path: Path) -> list[dict[str, Any]]:
    episodes = []
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                try:
                    episodes.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # a torn last line from an interrupted run
    return episodes


def _long(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [e for e in episodes if e["n_moves_reference"] >= H4_GATE.min_moves]


def load_runs(
    cfg: V1OODConfig, out: Path, arms: tuple[str, ...], seeds: tuple[int, ...]
) -> dict[tuple[str, int, str], dict[str, Any]]:
    info = read_json(out / "prepare.json")
    runs: dict[tuple[str, int, str], dict[str, Any]] = {}
    for arm in arms:
        for seed in seeds:
            for cond in cfg.conditions:
                d = run_dir(out, arm, seed) / cond.name
                episodes = _read_episodes(d / "eval.jsonl")
                expected = info["conditions"][cond.name]["kept"]
                runs[(arm, seed, cond.name)] = {
                    "episodes": episodes,
                    "n_expected": expected,
                    "complete": (d / "metrics.json").exists()
                    and len({e["problem_id"] for e in episodes}) >= expected,
                }
    return runs


def _paired_delta(
    a: dict[str, float], b: dict[str, float], rng: np.random.Generator
) -> dict[str, Any] | None:
    from .v1_report import _bootstrap

    ids = sorted(set(a) & set(b))
    if not ids:
        return None
    deltas = np.array([a[i] - b[i] for i in ids])
    lo, hi = _bootstrap(deltas, rng)
    return {
        "n": len(ids),
        "delta_pts": float(deltas.mean()) * 100.0,
        "ci_pts": [lo * 100.0, hi * 100.0],
    }


def reproduction(
    cfg: V1OODConfig,
    out: Path,
    source_root: Path,
    arm: str,
    seed: int,
) -> dict[str, Any] | None:
    """How closely the identity condition reproduces the saved V1 episodes."""
    saved = {
        e["problem_id"]: e
        for e in _read_episodes(source_dir(cfg, source_root, arm, seed) / "eval.jsonl")
    }
    fresh = {
        e["problem_id"]: e
        for e in _read_episodes(run_dir(out, arm, seed) / "identity" / "eval.jsonl")
    }
    ids = sorted(set(saved) & set(fresh))
    if not ids:
        return None
    return {
        "n": len(ids),
        "same_moves": float(
            np.mean([saved[i]["moves"] == fresh[i]["moves"] for i in ids])
        ),
        "same_success": float(
            np.mean([saved[i]["success"] == fresh[i]["success"] for i in ids])
        ),
        "success_saved": float(np.mean([saved[i]["success"] for i in ids])),
        "success_fresh": float(np.mean([fresh[i]["success"] for i in ids])),
    }


def report(
    cfg: V1OODConfig,
    out: Path,
    source_root: Path | None = None,
    arms: tuple[str, ...] = (ARM_TOKENS, ARM_TABLE),
    seeds: tuple[int, ...] = V1.seeds,
    seed: int = 0,
) -> tuple[dict[str, Any], str]:
    from .v1_report import _bootstrap, _per_problem

    rng = np.random.default_rng(seed)
    runs = load_runs(cfg, out, arms, seeds)
    info = read_json(out / "prepare.json")
    long_name = f">={H4_GATE.min_moves} moves"
    lines = [f"# H4 — V1 adapters under pose conditions ({out})", ""]
    lines.append(
        f"Adapters: V1 protocol {cfg.source_protocol}; eval set data-{cfg.data_key} "
        f"({info['n_source']} problems); ood {info['fingerprint']['ood']}."
    )
    lines.append("")
    lines.append(
        f"| condition | arm | problems {long_name} | success {long_name} [CI95] | "
        f"success all | wrong selection {long_name} | Δ vs identity {long_name} [CI95] |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    per_problem: dict[tuple[str, str], dict[str, float]] = {}
    rows: dict[str, Any] = {}
    for cond in cfg.conditions:
        for arm in arms:
            episodes = [e for s in seeds for e in runs[(arm, s, cond.name)]["episodes"]]
            long_eps = _long(episodes)
            pp = _per_problem(long_eps, "success") if long_eps else {}
            per_problem[(cond.name, arm)] = pp
            values = np.array(list(pp.values()))
            ci = _bootstrap(values, rng) if values.size else (math.nan, math.nan)
            row = {
                "n_long": len(pp),
                "success_long": float(values.mean()) if values.size else math.nan,
                "success_long_ci": list(ci),
                "success_all": float(
                    np.mean(list(_per_problem(episodes, "success").values()))
                )
                if episodes
                else math.nan,
                "selection_error_long": float(
                    np.mean(list(_per_problem(long_eps, "selection_error").values()))
                )
                if long_eps
                else math.nan,
            }
            vs = (
                _paired_delta(pp, per_problem[("identity", arm)], rng)
                if cond.name != "identity" and ("identity", arm) in per_problem
                else None
            )
            row["vs_identity"] = vs
            rows[f"{cond.name}/{arm}"] = row
            vs_text = (
                f"{vs['delta_pts']:+.1f} [{vs['ci_pts'][0]:+.1f}, {vs['ci_pts'][1]:+.1f}]"
                if vs
                else "—"
            )
            lines.append(
                f"| {cond.name} | {arm} | {row['n_long']} | "
                f"{row['success_long']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | "
                f"{row['success_all']:.3f} | {row['selection_error_long']:.3f} | "
                f"{vs_text} |"
            )
    lines.append("")
    lines.append(f"Paired Δsuccess tokens − table on {long_name} (absolute gap):")
    lines.append("")
    deltas: dict[str, Any] = {}
    for cond in cfg.conditions:
        d = _paired_delta(
            per_problem.get((cond.name, ARM_TOKENS), {}),
            per_problem.get((cond.name, ARM_TABLE), {}),
            rng,
        )
        deltas[cond.name] = d
        tag = " (far)" if cond.far else ""
        if d:
            lines.append(
                f"- {cond.name}{tag}: {d['delta_pts']:+.1f} pts, CI95 "
                f"[{d['ci_pts'][0]:+.1f}, {d['ci_pts'][1]:+.1f}] ({d['n']} problems; "
                f"{info['conditions'][cond.name]['rejected']} rejected at preparation)"
            )
        else:
            lines.append(f"- {cond.name}{tag}: no paired episodes")
    # Robustness, what the gate reads: per problem, how much less the tokens
    # arm drops from identity than the table arm does (positive = tokens more
    # robust). Paired over problems present in all four cells.
    lines.append("")
    lines.append(
        f"Robustness on {long_name}: (tokens − tokens@identity) − "
        "(table − table@identity), paired per problem:"
    )
    lines.append("")
    robust: dict[str, Any] = {}
    tok_id = per_problem.get(("identity", ARM_TOKENS), {})
    tab_id = per_problem.get(("identity", ARM_TABLE), {})
    for cond in cfg.conditions:
        if cond.name == "identity":
            continue
        tok = per_problem.get((cond.name, ARM_TOKENS), {})
        tab = per_problem.get((cond.name, ARM_TABLE), {})
        ids = set(tok) & set(tab) & set(tok_id) & set(tab_id)
        r = _paired_delta(
            {i: tok[i] - tok_id[i] for i in ids},
            {i: tab[i] - tab_id[i] for i in ids},
            rng,
        )
        robust[cond.name] = r
        tag = " (far, gated)" if cond.far else ""
        if r:
            lines.append(
                f"- {cond.name}{tag}: {r['delta_pts']:+.1f} pts, CI95 "
                f"[{r['ci_pts'][0]:+.1f}, {r['ci_pts'][1]:+.1f}] ({r['n']} problems)"
            )
        else:
            lines.append(f"- {cond.name}{tag}: no paired episodes")
    # identity must reproduce the saved V1 episodes, or nothing here is H1's model
    repro: dict[str, Any] = {}
    lines.append("")
    lines.append("Identity vs saved V1 episodes (same problems):")
    for arm in arms:
        for s in seeds:
            r = (
                reproduction(cfg, out, source_root, arm, s)
                if source_root is not None
                else None
            )
            repro[f"{arm}/s{s}"] = r
            if r:
                lines.append(
                    f"- {arm}/s{s}: {r['n']} problems, same moves {r['same_moves']:.3f}, "
                    f"same success {r['same_success']:.3f} (success saved "
                    f"{r['success_saved']:.3f}, now {r['success_fresh']:.3f})"
                )
            else:
                lines.append(f"- {arm}/s{s}: no saved or no fresh identity episodes")
    unreproduced = [
        k
        for k, r in repro.items()
        if r is None or r["same_success"] < H4_GATE.min_reproduction
    ]
    # gate
    missing = [
        f"{a}/s{s}/{c} ({len(r['episodes'])}/{r['n_expected']})"
        for (a, s, c), r in sorted(runs.items())
        if not r["complete"]
    ]
    far = [c.name for c in cfg.conditions if c.far]
    gate: dict[str, Any] = {
        "verdict": None,
        "incomplete": missing,
        "unreproduced": unreproduced,
        "far": far,
    }
    lines.append("")
    small = [c for c in far if not robust.get(c) or robust[c]["n"] < H4_GATE.n_min]
    if missing:
        gate["verdict"] = "withheld"
        lines.append(
            "H4 verdict: withheld — incomplete: " + ", ".join(missing[:6]) + "."
        )
    elif unreproduced:
        gate["verdict"] = "withheld"
        lines.append(
            "H4 verdict: withheld — identity does not reproduce the saved V1 "
            f"episodes (same success < {H4_GATE.min_reproduction}) for: "
            + ", ".join(unreproduced)
            + "."
        )
    elif small:
        gate["verdict"] = "withheld"
        lines.append(
            f"H4 verdict: withheld — fewer than {H4_GATE.n_min} paired long problems "
            f"on: {', '.join(small)}."
        )
    else:
        wins = [
            c
            for c in far
            if robust[c]["delta_pts"] >= H4_GATE.go_delta_pts
            and robust[c]["ci_pts"][0] > 0
        ]
        if wins:
            gate["verdict"], why = (
                "GO",
                "tokens more robust by at least the go margin on " + ", ".join(wins),
            )
        elif all(robust[c]["delta_pts"] <= H4_GATE.kill_delta_pts for c in far):
            gate["verdict"], why = (
                "KILL",
                "every far condition at/below the kill margin",
            )
        else:
            gate["verdict"], why = "RECONSIDER", "between the kill and go margins"
        lines.append(
            f"H4 verdict: {gate['verdict']} ({why}; robustness go ≥ "
            f"{H4_GATE.go_delta_pts:.0f} pts with CI > 0 on any far condition, "
            f"kill ≤ {H4_GATE.kill_delta_pts:.0f} pts on all)."
        )
    text = "\n".join(lines) + "\n"
    payload = {
        "out": str(out),
        "ood": info["fingerprint"]["ood"],
        "rows": rows,
        "deltas": deltas,
        "robustness": robust,
        "gate": gate,
        "reproduction": repro,
        "prepare": {
            k: {"kept": v["kept"], "rejected": v["rejected"]}
            for k, v in info["conditions"].items()
        },
    }
    return payload, text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--arm", choices=V1.arms, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="V1 runs and data (default ARTIFACT_ROOT/v1)",
    )
    ap.add_argument("--smoke", action="store_true", help="a few problems per bucket")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    cfg = V1_OOD
    out = args.out or default_out(cfg, args.smoke)
    source_root = args.source_root or default_source_root()
    if args.prepare:
        info = prepare(cfg, out, source_root, args.smoke)
        print(json.dumps({k: v["kept"] for k, v in info["conditions"].items()}))
    if args.eval:
        if args.arm is None or args.seed is None:
            ap.error("--eval needs --arm and --seed")
        result = evaluate_run(
            cfg, args.arm, args.seed, out, source_root, args.smoke, args.device
        )
        print(json.dumps(result["conditions"]), flush=True)
    if args.report:
        payload, text = report(cfg, out, source_root, seed=V1.eval_seed)
        write_json(out / "report.json", payload)
        (out / "report.md").write_text(text)
        sys.stdout.write(text)
    if not (args.prepare or args.eval or args.report):
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
