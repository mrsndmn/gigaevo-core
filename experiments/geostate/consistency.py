"""E0 — consistency of existing models under similarities and relabelling.

For every construction problem the probe makes K copies: the identity, and
copies that differ from it by a random orientation-preserving similarity
(translation, rotation, uniform scale, or all three), a relabelling of the
given points, or both. Each copy is a construction problem in its own right —
the target is re-derived by the kernel from the transformed given points, so
success is judged in the copy's own frame. Consistency is the rate at which a
model's answers to the copies agree once mapped back to the original frame.

Arms: `text` puts the coordinates in the prompt; `figure` renders the labelled
points on a grid (the benchmark's own dots style) and the prompt carries no
coordinates. The model's program is executed by the kernel; nothing the model
writes is evaluated as code.

    python -m experiments.geostate.consistency --config e0-synthetic-text --dry
    python -m experiments.geostate.consistency --config e0-synthetic-text --run --limit 10
    python -m experiments.geostate.consistency --config e0-synthetic-text --summary
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import itertools
import json
import math
from pathlib import Path
import random
import re
import sys
from typing import Any

import numpy as np

from experiments.noregeo_synth import figures as _figures

from . import dsl, prompts
from .cga import IDENTITY, Similarity
from .client import make_client, read_json, request, write_json
from .configs import (
    ARM_FIGURE,
    ARM_TEXT,
    DEFAULT_CONFIG,
    E0_GATE,
    GENERATOR,
    OUT_DIR,
    SET_AUX,
    SET_SYNTHETIC,
    TOL,
    E0Config,
    GroupSpec,
    ToleranceSpec,
    config_named,
    experiment_configs,
)
from .generator import Problem, build_problem, generate, load_problems, save_problems
from .macros import distance_between, inverse_params, object_from_params, relabel_expr
from .state import MoveError, same_geometry

# ---------------------------------------------------------------------------
# Copies
# ---------------------------------------------------------------------------


@dataclass
class Copy:
    index: int
    kind: str
    g: Similarity
    perm: dict[str, str]  # original label -> copy label
    problem: Problem

    @property
    def id(self) -> str:
        return f"c{self.index}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "kind": self.kind,
            "g": self.g.to_dict(),
            "perm": self.perm,
            "problem": self.problem.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Copy:
        return cls(
            index=int(d["index"]),
            kind=d["kind"],
            g=Similarity.from_dict(d["g"]),
            perm=dict(d["perm"]),
            problem=Problem.from_dict(d["problem"]),
        )


def sample_similarity(rng: random.Random, group: GroupSpec, kind: str) -> Similarity:
    tx = ty = 0.0
    theta = 0.0
    scale = 1.0
    if kind in ("translate", "similarity", "similarity+relabel"):
        tx = rng.uniform(-group.translation_max, group.translation_max)
        ty = rng.uniform(-group.translation_max, group.translation_max)
    if kind in ("rotate", "similarity", "similarity+relabel"):
        theta = rng.uniform(0.0, 2.0 * math.pi)
    if kind in ("scale", "similarity", "similarity+relabel"):
        scale = math.exp(
            rng.uniform(math.log(group.scale_min), math.log(group.scale_max))
        )
    reflect = kind == "reflect" and group.reflection
    return Similarity(tx=tx, ty=ty, theta=theta, scale=scale, reflect=reflect)


def sample_perm(rng: random.Random, labels: list[str], kind: str) -> dict[str, str]:
    if kind not in ("relabel", "similarity+relabel") or len(labels) < 2:
        return {label: label for label in labels}
    shuffled = list(labels)
    while shuffled == labels:
        rng.shuffle(shuffled)
    return dict(zip(labels, shuffled, strict=True))


def _round2(v: float) -> float:
    return float(f"{v:.2f}")


def transformed_problem(
    problem: Problem,
    g: Similarity,
    perm: dict[str, str],
    tol: ToleranceSpec,
    index: int,
) -> Problem:
    """The same construction problem in another frame and with other labels.

    Given coordinates are rounded to two decimals — they are what the model
    reads — and the target is re-derived from them, so the copy is exact.
    """
    given = {
        perm[label]: tuple(_round2(c) for c in g.xy(x, y))
        for label, (x, y) in problem.given.items()
    }
    # keep the label order of the original alphabet so prompts read naturally
    given = dict(sorted(given.items()))
    expr = relabel_expr(problem.expr, perm)
    return build_problem(
        f"{problem.id}/c{index}",
        problem.seed,
        given,  # type: ignore[arg-type]
        expr,
        tol,
        source=problem.source,
        meta={**problem.meta, "copy_of": problem.id},
    )


def make_copies(
    problem: Problem, cfg: E0Config, rng: random.Random, tol: ToleranceSpec = TOL
) -> list[Copy]:
    kinds = [k for k in cfg.kinds if k != "identity"]
    copies = [
        Copy(
            0, "identity", IDENTITY, {label: label for label in problem.given}, problem
        )
    ]
    for index in range(1, cfg.copies):
        kind = kinds[(index - 1) % len(kinds)]
        for _attempt in range(20):
            g = sample_similarity(rng, cfg.group, kind)
            perm = sample_perm(rng, list(problem.given), kind)
            try:
                copy_problem = transformed_problem(problem, g, perm, tol, index)
            except (MoveError, ValueError):
                continue
            copies.append(Copy(index, kind, g, perm, copy_problem))
            break
    return copies


# ---------------------------------------------------------------------------
# Problem sets and preparation
# ---------------------------------------------------------------------------


def problems_for(
    cfg: E0Config, tol: ToleranceSpec = TOL
) -> tuple[list[Problem], list[Any]]:
    if cfg.problem_set == SET_SYNTHETIC:
        return generate(GENERATOR, tol, cfg.seed, cfg.per_bucket), []
    if cfg.problem_set == SET_AUX:
        from .aux import load_aux_problems

        return load_aux_problems(cfg.per_stem, cfg.seed, tol)
    raise ValueError(f"unknown problem set {cfg.problem_set!r}")


def prepare(
    cfg: E0Config, out: Path, tol: ToleranceSpec = TOL
) -> list[tuple[Problem, list[Copy]]]:
    """Freeze problems and copies under `out` (idempotent: reloads if present)."""
    out.mkdir(parents=True, exist_ok=True)
    problems_path = out / "problems.jsonl"
    copies_path = out / "copies.jsonl"
    if problems_path.exists() and copies_path.exists():
        problems = load_problems(problems_path)
        by_problem: dict[str, list[Copy]] = defaultdict(list)
        for line in copies_path.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                by_problem[d["problem_id"]].append(Copy.from_dict(d["copy"]))
        return [(p, by_problem[p.id]) for p in problems]
    problems, skipped = problems_for(cfg, tol)
    rng = random.Random(cfg.seed)
    prepared = [(p, make_copies(p, cfg, rng, tol)) for p in problems]
    save_problems(problems, problems_path)
    tmp = copies_path.with_suffix(".tmp")
    with tmp.open("w") as fh:
        for p, copies in prepared:
            for c in copies:
                fh.write(json.dumps({"problem_id": p.id, "copy": c.to_dict()}) + "\n")
    tmp.replace(copies_path)
    write_json(
        out / "prepare.json",
        {
            "config": cfg.name,
            "n_problems": len(problems),
            "n_copies": sum(len(c) for _, c in prepared),
            "skipped": [s.__dict__ for s in skipped],
        },
    )
    return prepared


# ---------------------------------------------------------------------------
# One measurement
# ---------------------------------------------------------------------------


def model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)


def job_dir(out: Path, model: str, problem: Problem, copy: Copy) -> Path:
    return out / model_slug(model) / problem.id / copy.id


def figure_for(copy: Copy, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "figure.png"
    if not path.exists():
        points = {label: [x, y] for label, (x, y) in copy.problem.given.items()}
        _figures.render_dots(
            points, "Given points", path, view=_figures.view_box_for(points)
        )
    return path


def messages_for(copy: Copy, arm: str, directory: Path) -> list[dict[str, Any]]:
    if arm == ARM_TEXT:
        return prompts.messages_text(copy.problem.given, copy.problem.description)
    if arm == ARM_FIGURE:
        return prompts.messages_figure(
            list(copy.problem.given),
            copy.problem.description,
            figure_for(copy, directory),
        )
    raise ValueError(f"unknown arm {arm!r}")


def evaluate(
    copy: Copy, content: str | None, tol: ToleranceSpec = TOL
) -> dict[str, Any]:
    """Execute the model's program on the copy's state and score it."""
    problem = copy.problem
    if content is None:
        return {
            "success": False,
            "answer_kind": None,
            "answer_original_frame": None,
            "error_kind": "request_error",
            "n_moves": 0,
        }
    ex = dsl.run_text(content, problem.state(tol))
    target = object_from_params(problem.target_kind, problem.target_params)
    scale = ex.state.scale
    success = bool(
        ex.answer is not None
        and same_geometry(ex.answer, target, scale, tol.coincidence)
    )
    lenient = any(
        same_geometry(o, target, scale, tol.coincidence) for o in ex.state.objects
    )
    record: dict[str, Any] = {
        "success": success,
        "target_reached": lenient,
        "answer_kind": ex.answer.kind if ex.answer is not None else None,
        "answer_original_frame": (
            list(inverse_params(ex.answer.kind, ex.answer.params(), copy.g))
            if ex.answer is not None
            else None
        ),
        "error_kind": ex.error_kind,
        "error": ex.error,
        "n_moves": ex.n_moves,
        "n_objects": len(ex.state.objects),
    }
    return record


def agree(a: dict[str, Any], b: dict[str, Any], scale: float, rel_tol: float) -> bool:
    """Do two copies' outcomes agree, in the original frame?"""
    if a["success"] and b["success"]:
        return True
    if a["success"] != b["success"]:
        return False
    pa, pb = a.get("answer_original_frame"), b.get("answer_original_frame")
    if pa is None or pb is None:
        return pa is None and pb is None and a.get("error_kind") == b.get("error_kind")
    if a["answer_kind"] != b["answer_kind"]:
        return False
    return distance_between(a["answer_kind"], pa, pb) <= rel_tol * scale


def pairwise_agreement(
    records: list[dict[str, Any]], scale: float, rel_tol: float
) -> float | None:
    pairs = list(itertools.combinations(records, 2))
    if not pairs:
        return None
    return sum(agree(a, b, scale, rel_tol) for a, b in pairs) / len(pairs)


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def run_one(
    client,
    cfg: E0Config,
    model: str,
    problem: Problem,
    copy: Copy,
    out: Path,
    tol: ToleranceSpec,
) -> dict[str, Any]:
    directory = job_dir(out, model, problem, copy)
    eval_path = directory / "eval.json"
    if eval_path.exists():
        return read_json(eval_path)
    messages = messages_for(copy, cfg.arm, directory)
    result = request(
        client, cfg, model, messages, directory, salt=f"{problem.id}/{copy.id}"
    )
    record = evaluate(copy, result.get("content"), tol)
    record.update(
        {
            "model": model,
            "problem_id": problem.id,
            "copy": copy.id,
            "kind": copy.kind,
            "bucket": problem.bucket,
            "n_moves_reference": problem.n_moves,
            "request_error": result.get("error"),
            "usage": result.get("usage", {}),
        }
    )
    write_json(eval_path, record)
    return record


def run(
    cfg: E0Config,
    out: Path,
    limit: int | None = None,
    workers: int | None = None,
    tol: ToleranceSpec = TOL,
) -> int:
    prepared = prepare(cfg, out, tol)
    if limit is not None:
        prepared = prepared[:limit]
    client, creds = make_client(cfg)
    print(
        f"[{cfg.name}] credentials from {creds.source}; {len(prepared)} problems x {cfg.copies} copies x {len(cfg.models)} models",
        flush=True,
    )
    jobs = [
        (model, p, c) for model in cfg.models for p, copies in prepared for c in copies
    ]
    failures = 0
    with ThreadPoolExecutor(max_workers=workers or cfg.workers) as pool:
        futures = {
            pool.submit(run_one, client, cfg, m, p, c, out, tol): (m, p, c)
            for m, p, c in jobs
        }
        for i, fut in enumerate(as_completed(futures), start=1):
            m, p, c = futures[fut]
            try:
                record = fut.result()
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"  ! {m} {p.id}/{c.id}: {type(exc).__name__}: {exc}", flush=True)
                continue
            if record.get("request_error"):
                failures += 1
            if i % 25 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} done", flush=True)
    print(f"[{cfg.name}] finished; {failures} failed requests", flush=True)
    return 0 if failures == 0 else 3


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _bootstrap_ci(
    values: list[float], rng: np.random.Generator, n: int
) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return math.nan, math.nan
    if arr.size == 1:
        return float(arr[0]), float(arr[0])
    means = np.array(
        [rng.choice(arr, size=arr.size, replace=True).mean() for _ in range(n)]
    )
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def collect(cfg: E0Config, out: Path, tol: ToleranceSpec = TOL) -> dict[str, Any]:
    prepared = prepare(cfg, out, tol)
    rows: list[dict[str, Any]] = []
    for model in cfg.models:
        for problem, copies in prepared:
            records = []
            for copy in copies:
                path = job_dir(out, model, problem, copy) / "eval.json"
                if path.exists():
                    rec = read_json(path)
                    records.append(rec)
            if not records:
                continue
            scale = problem.state(tol).scale
            identity = next((r for r in records if r["copy"] == "c0"), None)
            rows.append(
                {
                    "model": model,
                    "problem_id": problem.id,
                    "bucket": problem.bucket,
                    "n_moves": problem.n_moves,
                    "n_copies": len(records),
                    "consistency": pairwise_agreement(records, scale, tol.agreement),
                    "success_identity": identity["success"] if identity else None,
                    "success_mean": sum(r["success"] for r in records) / len(records),
                    "vs_identity": (
                        sum(
                            agree(identity, r, scale, tol.agreement)
                            for r in records
                            if r is not identity
                        )
                        / max(1, len(records) - 1)
                        if identity
                        else None
                    ),
                    "by_kind": {r["kind"]: r["success"] for r in records},
                    "error_kinds": [r.get("error_kind") for r in records],
                }
            )
    return {"config": cfg.name, "rows": rows}


def summarize(
    cfg: E0Config, out: Path, tol: ToleranceSpec = TOL
) -> tuple[dict[str, Any], str]:
    data = collect(cfg, out, tol)
    rng = np.random.default_rng(cfg.seed)
    lines = [f"# E0 {cfg.name} — arm {cfg.arm}, set {cfg.problem_set}", ""]
    lines.append(
        "| model | bucket | n | success (identity) | success (all copies) | consistency | CI95 | vs identity |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    gate: dict[str, Any] = {}
    summary: list[dict[str, Any]] = []
    for model in cfg.models:
        rows = [r for r in data["rows"] if r["model"] == model]
        buckets = sorted({r["bucket"] for r in rows}) + [
            "all",
            f">={E0_GATE.min_moves} moves",
        ]
        for bucket in buckets:
            if bucket == "all":
                sel = rows
            elif bucket.startswith(">="):
                sel = [r for r in rows if r["n_moves"] >= E0_GATE.min_moves]
            else:
                sel = [r for r in rows if r["bucket"] == bucket]
            cons = [r["consistency"] for r in sel if r["consistency"] is not None]
            if not sel:
                continue
            succ_id = [
                float(r["success_identity"])
                for r in sel
                if r["success_identity"] is not None
            ]
            succ_all = [r["success_mean"] for r in sel]
            vs_id = [r["vs_identity"] for r in sel if r["vs_identity"] is not None]
            lo, hi = _bootstrap_ci(cons, rng, cfg.bootstrap)
            entry = {
                "model": model,
                "bucket": bucket,
                "n": len(sel),
                "success_identity": float(np.mean(succ_id)) if succ_id else math.nan,
                "success_all": float(np.mean(succ_all)),
                "consistency": float(np.mean(cons)) if cons else math.nan,
                "consistency_ci": [lo, hi],
                "vs_identity": float(np.mean(vs_id)) if vs_id else math.nan,
            }
            summary.append(entry)
            lines.append(
                f"| {model} | {bucket} | {entry['n']} | {entry['success_identity']:.3f} | "
                f"{entry['success_all']:.3f} | {entry['consistency']:.3f} | "
                f"[{lo:.3f}, {hi:.3f}] | {entry['vs_identity']:.3f} |"
            )
            if bucket.startswith(">="):
                gate[model] = entry["consistency"]
    # success by transform kind
    lines.append("")
    lines.append("| model | kind | n | success |")
    lines.append("|---|---|---|---|")
    by_kind: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in data["rows"]:
        for kind, s in r["by_kind"].items():
            by_kind[(r["model"], kind)].append(float(s))
    for (model, kind), vals in sorted(by_kind.items()):
        lines.append(f"| {model} | {kind} | {len(vals)} | {np.mean(vals):.3f} |")
    # kill gate
    lines.append("")
    if gate:
        best_model = max(gate, key=lambda m: gate[m] if not math.isnan(gate[m]) else -1)
        best = gate[best_model]
        verdict = (
            "H0 FALSIFIED (consistency at/above the kill gate)"
            if best >= E0_GATE.consistency_kill
            else "H0 stands (inconsistency below the kill gate)"
        )
        lines.append(
            f"Kill gate (problems with >= {E0_GATE.min_moves} moves, best open model "
            f"{best_model}): consistency {best:.3f} vs {E0_GATE.consistency_kill:.2f} -> {verdict}"
        )
    text = "\n".join(lines) + "\n"
    payload = {
        "config": cfg.name,
        "summary": summary,
        "gate": gate,
        "rows": data["rows"],
    }
    return payload, text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=", ".join(c.name for c in experiment_configs),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"output root (default {OUT_DIR}/<config>)",
    )
    ap.add_argument(
        "--dry", action="store_true", help="prepare problems and copies; no requests"
    )
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="only the first N problems")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args(argv)
    cfg = config_named(args.config)
    out = args.out or (OUT_DIR / cfg.name)
    code = 0
    if args.dry:
        prepared = prepare(cfg, out)
        n_copies = sum(len(c) for _, c in prepared)
        kinds: dict[str, int] = defaultdict(int)
        buckets: dict[str, int] = defaultdict(int)
        for p, copies in prepared:
            buckets[p.bucket] += 1
            for c in copies:
                kinds[c.kind] += 1
        print(
            f"[{cfg.name}] {len(prepared)} problems ({dict(buckets)}), {n_copies} copies ({dict(kinds)}), "
            f"{n_copies * len(cfg.models)} requests over {len(cfg.models)} model(s); out={out}"
        )
    if args.run:
        code = run(cfg, out, limit=args.limit, workers=args.workers)
    if args.summary:
        payload, text = summarize(cfg, out)
        write_json(out / "summary.json", payload)
        (out / "summary.md").write_text(text)
        sys.stdout.write(text)
    if not (args.dry or args.run or args.summary):
        ap.print_help()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
