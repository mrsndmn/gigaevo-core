"""E0 — consistency of existing models under similarities and relabelling.

For every construction problem the probe makes K copies: the identity, and
copies that differ from it by a random orientation-preserving similarity
(translation, rotation, uniform scale, or all three), a relabelling of the
given points, or both. Each copy is a construction problem in its own right —
the target is re-derived by the kernel from the transformed given points, so
success is judged in the copy's own frame. Consistency is the rate at which a
model's *answers* to the copies agree once mapped back to the original frame;
copies without a parseable answer, and copies whose request failed, are
reported as coverage and never counted as agreement.

Arms: `text` puts the coordinates in the prompt; `figure` renders the labelled
points on a grid (the benchmark's own dots style) and the prompt carries no
coordinates. The model's program is executed by the kernel; nothing the model
writes is evaluated as code.

    python -m experiments.geostate.consistency --config e0-synthetic-text --dry
    python -m experiments.geostate.consistency --config e0-synthetic-text --run --limit 12
    python -m experiments.geostate.consistency --config e0-synthetic-text --summary
    python -m experiments.geostate.consistency --config e0-synthetic-text --recover-errors

`--recover-errors` archives the request directories whose result is a
transport/auth error (for example after a token renewal) so that the next
`--run` re-issues exactly those requests; it refuses to touch a run that holds
the lock.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
import fcntl
import hashlib
import inspect
import itertools
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
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
    LENGTH_BUCKETS,
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
from .generator import (
    Problem,
    build_problem,
    generate_with_stats,
    goal_holds,
    load_problems,
    relabel_goal,
    save_problems,
)
from .macros import (
    distance_between,
    inverse_params,
    object_from_params,
    relabel_expr,
)
from .state import ERR_DEGENERATE, MoveError

REQUEST_ERROR = "request_error"
LOCK_NAME = ".lock"

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
    reflect = False
    if kind in ("translate", "similarity", "similarity+relabel"):
        tx = rng.uniform(-group.translation_max, group.translation_max)
        ty = rng.uniform(-group.translation_max, group.translation_max)
    if kind in (
        "rotate",
        "similarity",
        "similarity+relabel",
        "reflect",
        "reflect+relabel",
    ):
        theta = rng.uniform(0.0, 2.0 * math.pi)
    if kind in ("scale", "similarity", "similarity+relabel"):
        scale = math.exp(
            rng.uniform(math.log(group.scale_min), math.log(group.scale_max))
        )
    if kind in ("reflect", "reflect+relabel"):
        if not group.reflection:
            raise ValueError(f"kind {kind!r} needs a group with reflection=True")
        reflect = True
    return Similarity(tx=tx, ty=ty, theta=theta, scale=scale, reflect=reflect)


def sample_perm(rng: random.Random, labels: list[str], kind: str) -> dict[str, str]:
    if "relabel" not in kind or len(labels) < 2:
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
    reads — and the target is re-derived from them, so the copy is exact. A
    copy whose target, mapped back to the original frame, has drifted by more
    than the agreement tolerance (an ill-conditioned construction, or an
    orientation-dependent one under a reflection) is rejected.
    """
    given = {
        perm[label]: tuple(_round2(c) for c in g.xy(x, y))
        for label, (x, y) in problem.given.items()
    }
    # keep the label order of the original alphabet so prompts read naturally
    given = dict(sorted(given.items()))
    expr = relabel_expr(problem.expr, perm)
    copy = build_problem(
        f"{problem.id}/c{index}",
        problem.seed,
        given,  # type: ignore[arg-type]
        expr,
        tol,
        source=problem.source,
        meta={**problem.meta, "copy_of": problem.id},
        goal=relabel_goal(problem.goal, perm),
    )
    # The copy is a faithful transform of the problem iff its reference
    # answer, mapped back to the original frame, still solves the *original*
    # problem — within the agreement tolerance, because the given coordinates
    # were rounded. For coincidence goals this is the target drift check; for
    # predicate goals it accepts any valid answer (the "first" intersection
    # may legitimately be the other one after a reflection).
    original = problem.state(tol)
    back = object_from_params(
        copy.target_kind, inverse_params(copy.target_kind, copy.target_params, g)
    )
    loose = replace(
        tol, coincidence=tol.agreement, incidence=tol.agreement, tangency=tol.agreement
    )
    if not goal_holds(problem, original, back, loose):
        raise MoveError(ERR_DEGENERATE, "copy target drifts from the original")
    return copy


def make_copies(
    problem: Problem, cfg: E0Config, rng: random.Random, tol: ToleranceSpec = TOL
) -> list[Copy]:
    """The identity copy at index 0, then the other kinds in rotation.

    Two copies of one problem never coincide (a two-point problem has only one
    relabelling); when a kind cannot produce a fresh copy it falls back to the
    full similarity of the same relabelling class.
    """
    kinds = [k for k in cfg.kinds if k != "identity"]
    copies = [
        Copy(
            0, "identity", IDENTITY, {label: label for label in problem.given}, problem
        )
    ]
    seen = {
        json.dumps(problem.to_dict()["given"], sort_keys=True) + problem.description
    }
    for index in range(1, cfg.copies):
        kind = kinds[(index - 1) % len(kinds)]
        fallback = "similarity+relabel" if "relabel" in kind else "similarity"
        if "reflect" in kind:
            fallback = kind
        for attempt in range(40):
            use = kind if attempt < 20 else fallback
            g = sample_similarity(rng, cfg.group, use)
            perm = sample_perm(rng, list(problem.given), use)
            try:
                copy_problem = transformed_problem(problem, g, perm, tol, index)
            except (MoveError, ValueError):
                continue
            key = (
                json.dumps(copy_problem.to_dict()["given"], sort_keys=True)
                + copy_problem.description
            )
            if key in seen:
                continue
            seen.add(key)
            copies.append(Copy(index, use, g, perm, copy_problem))
            break
    return copies


# ---------------------------------------------------------------------------
# Problem sets and preparation
# ---------------------------------------------------------------------------


def problems_for(
    cfg: E0Config, tol: ToleranceSpec = TOL
) -> tuple[list[Problem], dict[str, Any]]:
    if cfg.problem_set == SET_SYNTHETIC:
        problems, stats = generate_with_stats(GENERATOR, tol, cfg.seed, cfg.per_bucket)
        if stats.underfilled:
            raise RuntimeError(f"generator underfilled buckets: {stats.underfilled}")
        return problems, {"generator": stats.to_dict()}
    if cfg.problem_set == SET_AUX:
        from .aux import load_aux_problems

        problems, skipped = load_aux_problems(cfg.per_stem, cfg.seed, tol)
        return problems, {"skipped": [asdict(s) for s in skipped]}
    raise ValueError(f"unknown problem set {cfg.problem_set!r}")


def protocol_hash(cfg: E0Config) -> str:
    """Everything that shapes a model's answer besides the problem itself.

    Prompt templates (the source of `prompts.py`) and the sampling parameters.
    Answers produced under different protocols are never summarised together:
    the hash is part of the run fingerprint and of the launcher's artifact
    directory name. The model list is *not* part of it — models live in
    separate directories and may come and go.
    """
    from . import cga, generator, macros, state

    material = json.dumps(
        {
            "prompts": inspect.getsource(prompts),
            "problems": [inspect.getsource(m) for m in (cga, state, macros, generator)],
            "generator": asdict(GENERATOR),
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "max_tokens": cfg.max_tokens,
            "reasoning_effort": cfg.reasoning_effort,
            "arm": cfg.arm,
        },
        sort_keys=True,
    )
    return hashlib.sha256(material.encode()).hexdigest()[:12]


def fingerprint(cfg: E0Config, tol: ToleranceSpec = TOL) -> dict[str, Any]:
    """What the frozen inputs depend on; checked when a run directory is reused."""
    return {
        "protocol": protocol_hash(cfg),
        "name": cfg.name,
        "problem_set": cfg.problem_set,
        "seed": cfg.seed,
        "copies": cfg.copies,
        "per_bucket": cfg.per_bucket,
        "per_stem": cfg.per_stem,
        "kinds": list(cfg.kinds),
        "group": asdict(cfg.group),
        "generator": asdict(GENERATOR),
        "tolerances": asdict(tol),
    }


def prepare(
    cfg: E0Config, out: Path, tol: ToleranceSpec = TOL, strict: bool = True
) -> list[tuple[Problem, list[Copy]]]:
    """Freeze problems and copies under `out` (idempotent: reloads if present).

    With `strict` (runs), frozen inputs prepared under another fingerprint are
    refused; read-only consumers (the summary) pass `strict=False` and get the
    frozen inputs as they are, whatever the current code says.
    """
    out.mkdir(parents=True, exist_ok=True)
    problems_path = out / "problems.jsonl"
    copies_path = out / "copies.jsonl"
    manifest_path = out / "prepare.json"
    if problems_path.exists() and copies_path.exists():
        manifest = read_json(manifest_path) if manifest_path.exists() else {}
        if strict and manifest.get("fingerprint") != fingerprint(cfg, tol):
            raise RuntimeError(
                f"frozen inputs at {out} were prepared for a different config; "
                "use another --out"
            )
        problems = load_problems(problems_path)
        by_problem: dict[str, list[Copy]] = defaultdict(list)
        for line in copies_path.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                by_problem[d["problem_id"]].append(Copy.from_dict(d["copy"]))
        return [(p, by_problem[p.id]) for p in problems]
    problems, extra = problems_for(cfg, tol)
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
        manifest_path,
        {
            "config": cfg.name,
            "fingerprint": fingerprint(cfg, tol),
            "n_problems": len(problems),
            "n_copies": sum(len(c) for _, c in prepared),
            **extra,
        },
    )
    return prepared


def stratified(
    prepared: list[tuple[Problem, list[Copy]]], limit: int | None
) -> list[tuple[Problem, list[Copy]]]:
    """The first `limit` problems taken round-robin across buckets."""
    if limit is None or limit >= len(prepared):
        return prepared
    queues: dict[str, list[tuple[Problem, list[Copy]]]] = defaultdict(list)
    for item in prepared:
        queues[item[0].bucket].append(item)
    order = [name for name, _, _ in LENGTH_BUCKETS if name in queues]
    order += [b for b in queues if b not in order]
    out: list[tuple[Problem, list[Copy]]] = []
    while len(out) < limit and any(queues.values()):
        for bucket in order:
            if queues[bucket] and len(out) < limit:
                out.append(queues[bucket].pop(0))
    return out


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
            "answered": False,
            "answer_kind": None,
            "answer_original_frame": None,
            "error_kind": REQUEST_ERROR,
            "n_moves": 0,
        }
    ex = dsl.run_text(content, problem.state(tol))
    success = bool(
        ex.answer is not None and goal_holds(problem, ex.state, ex.answer, tol)
    )
    lenient = any(goal_holds(problem, ex.state, o, tol) for o in ex.state.objects)
    return {
        "scorer": dsl.scorer_hash(),
        "success": success,
        "answered": ex.answer is not None,
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


def agree(
    a: dict[str, Any], b: dict[str, Any], scale: float, rel_tol: float
) -> bool | None:
    """Do two answered copies agree in the original frame? None if not both answered."""
    if not (a.get("answered") and b.get("answered")):
        return None
    if a["success"] and b["success"]:
        return True
    if a["success"] != b["success"]:
        return False
    if a["answer_kind"] != b["answer_kind"]:
        return False
    pa, pb = a["answer_original_frame"], b["answer_original_frame"]
    return distance_between(a["answer_kind"], pa, pb, scale) <= rel_tol * scale


def pairwise_agreement(
    records: list[dict[str, Any]], scale: float, rel_tol: float
) -> tuple[float | None, int]:
    """Mean agreement over answered pairs, and how many pairs that was."""
    verdicts = [
        v
        for a, b in itertools.combinations(records, 2)
        if (v := agree(a, b, scale, rel_tol)) is not None
    ]
    if not verdicts:
        return None, 0
    return sum(verdicts) / len(verdicts), len(verdicts)


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


class Clients:
    """One client per run, rebuilt when the token is about to expire."""

    def __init__(
        self, cfg: E0Config, ca_dir: Path | None = None, margin_s: float = 120.0
    ):
        self.cfg = cfg
        self.ca_dir = ca_dir
        self.margin_s = margin_s
        self._client = None
        self._creds = None

    def get(self):
        expires = getattr(self._creds, "expires_at", None)
        if self._client is None or (
            expires is not None and expires - time.time() < self.margin_s
        ):
            self._client, self._creds = make_client(self.cfg, self.ca_dir)
        return self._client

    @property
    def source(self) -> str:
        self.get()
        return getattr(self._creds, "source", "?")


def run_one(
    clients: Clients,
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
        clients.get(), cfg, model, messages, directory, salt=f"{problem.id}/{copy.id}"
    )
    meta = {
        "model": model,
        "problem_id": problem.id,
        "copy": copy.id,
        "kind": copy.kind,
        "bucket": problem.bucket,
        "n_moves_reference": problem.n_moves,
        "usage": result.get("usage", {}),
    }
    if result.get("error"):
        # Not persisted as an evaluation: the summary reports it as missing and
        # `--recover-errors` lets the next run re-issue it.
        return {**evaluate(copy, None, tol), **meta, REQUEST_ERROR: result["error"]}
    record = {**evaluate(copy, result.get("content"), tol), **meta}
    write_json(eval_path, record)
    return record


class RunLock:
    """An advisory `flock` on `<out>/.lock`, released by the kernel on death.

    A run stopped by SIGTERM (an MLS stop or preemption) therefore never
    leaves a lock behind that would block the next submission; the file
    itself is only a record of who last held it.
    """

    def __init__(self, out: Path):
        self.path = out / LOCK_NAME
        self._fh = None

    def __enter__(self) -> RunLock:
        self._fh = self.path.open("a+")
        try:
            fcntl.flock(self._fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._fh.close()
            self._fh = None
            raise RuntimeError(
                f"{self.path}: another run owns {self.path.parent}"
            ) from None
        self._fh.seek(0)
        self._fh.truncate()
        self._fh.write(json.dumps({"pid": os.getpid(), "started": time.time()}))
        self._fh.flush()
        return self

    def __exit__(self, *exc) -> None:
        if self._fh is not None:
            fcntl.flock(self._fh, fcntl.LOCK_UN)
            self._fh.close()
            self._fh = None


def lock_is_held(out: Path) -> bool:
    """Is a run alive on `out`? (Tries to take the flock and gives it back.)"""
    path = out / LOCK_NAME
    if not path.exists():
        return False
    with path.open("a+") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        fcntl.flock(fh, fcntl.LOCK_UN)
    return False


class EndpointUnavailable(RuntimeError):
    """The model listing failed: nothing is scheduled (exit 4)."""


def served_models(client) -> set[str]:
    """Model ids the endpoint serves right now; raises if the listing fails.

    A listing that cannot be fetched means every request would fail the same
    way, so the run aborts before scheduling anything rather than burning the
    retry ladder on thousands of copies.
    """
    try:
        return {m.id for m in client.models.list().data}
    except Exception as exc:  # noqa: BLE001 - reported, then fatal
        raise EndpointUnavailable(
            f"could not list models: {type(exc).__name__}: {exc}"
        ) from exc


EXIT_FAILED_REQUESTS = 3
EXIT_ENDPOINT_UNAVAILABLE = 4
EXIT_MODELS_ABSENT = 5


def _model_gone(record: dict[str, Any]) -> bool:
    error = str(record.get(REQUEST_ERROR) or "")
    return "NotFoundError" in error and "model" in error.lower()


def run(
    cfg: E0Config,
    out: Path,
    limit: int | None = None,
    workers: int | None = None,
    tol: ToleranceSpec = TOL,
) -> int:
    """Issue every missing request; exit 0 only when every configured model ran clean.

    Exit codes: 3 = some requests failed (recoverable), 4 = the endpoint could
    not even list its models (nothing scheduled), 5 = a configured model was
    not served (skipped) or went away mid-run.
    """
    prepared = stratified(prepare(cfg, out, tol), limit)
    with RunLock(out):
        clients = Clients(cfg, out)
        # The roster changes: a model can be unloaded between runs or during
        # one. Skip the absent ones instead of caching a 404 per copy; a later
        # run picks them up when they are back.
        try:
            served = served_models(clients.get())
        except EndpointUnavailable as exc:
            write_json(
                out / "run.json",
                {"served": None, "error": str(exc), "started": time.time()},
            )
            print(f"[{cfg.name}] {exc}; nothing scheduled", flush=True)
            return EXIT_ENDPOINT_UNAVAILABLE
        absent = [m for m in cfg.models if m not in served]
        models = [m for m in cfg.models if m in served]
        if absent:
            print(
                f"[{cfg.name}] not served now, skipped: {', '.join(absent)}", flush=True
            )
        write_json(
            out / "run.json",
            {"served": sorted(served), "skipped": absent, "started": time.time()},
        )
        print(
            f"[{cfg.name}] credentials from {clients.source}; {len(prepared)} problems "
            f"x {cfg.copies} copies x {len(models)} models",
            flush=True,
        )
        jobs = [(m, p, c) for m in models for p, copies in prepared for c in copies]
        failures = 0
        gone: set[str] = set()

        def work(m: str, p: Problem, c: Copy) -> dict[str, Any] | None:
            if m in gone:
                return None  # unloaded mid-run: leave the copy unrequested
            record = run_one(clients, cfg, m, p, c, out, tol)
            if _model_gone(record):
                gone.add(m)
            return record

        with ThreadPoolExecutor(max_workers=workers or cfg.workers) as pool:
            futures = {pool.submit(work, m, p, c): (m, p, c) for m, p, c in jobs}
            for i, fut in enumerate(as_completed(futures), start=1):
                m, p, c = futures[fut]
                try:
                    record = fut.result()
                except Exception as exc:  # noqa: BLE001
                    failures += 1
                    print(
                        f"  ! {m} {p.id}/{c.id}: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    continue
                if record is not None and record.get(REQUEST_ERROR):
                    failures += 1
                if i % 25 == 0 or i == len(jobs):
                    print(f"  {i}/{len(jobs)} done", flush=True)
        if gone:
            print(
                f"[{cfg.name}] unloaded mid-run, remaining copies left unrequested: {', '.join(sorted(gone))}",
                flush=True,
            )
        write_json(
            out / "run.json",
            {
                "served": sorted(served),
                "skipped": absent,
                "gone_mid_run": sorted(gone),
                "failed_requests": failures,
                "started": None,
                "finished": time.time(),
            },
        )
        print(f"[{cfg.name}] finished; {failures} failed requests", flush=True)
        if failures:
            return EXIT_FAILED_REQUESTS
        if absent or gone:
            return EXIT_MODELS_ABSENT
        return 0


COPY_DIR_RE = re.compile(r"c\d+")


def rescore(
    cfg: E0Config, out: Path, tol: ToleranceSpec = TOL, force: bool = False
) -> int:
    """Re-evaluate every cached model output with the current parser/scorer.

    The model side (prompt, sampling) is untouched: only the offline scoring
    of the cached `result.json` changes, so this is legitimate on a frozen
    run of any protocol — the case that motivated it is a stricter parser
    having counted unambiguous programs as syntax errors. Refuses a live run.
    """
    if lock_is_held(out) and not force:
        # Safe in principle (every write is atomic and the worker never
        # rewrites an existing eval.json), but say so explicitly.
        raise RuntimeError(
            f"{out / LOCK_NAME} is held: pass --force to rescore a live run"
        )
    prepared = prepare(cfg, out, tol, strict=False)
    n = 0
    for model in cfg.models:
        for problem, copies in prepared:
            for copy in copies:
                directory = job_dir(out, model, problem, copy)
                result_path = directory / "result.json"
                if not result_path.exists():
                    continue
                if force and time.time() - result_path.stat().st_mtime < 120:
                    continue  # the worker may still be writing this copy's files
                result = read_json(result_path)
                if result.get("error"):
                    continue
                old = (
                    read_json(directory / "eval.json")
                    if (directory / "eval.json").exists()
                    else {}
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
                        "usage": result.get("usage", {}),
                        "rescored": True,
                    }
                )
                if old.get("error_kind") != record.get("error_kind") or old.get(
                    "success"
                ) != record.get("success"):
                    n += 1
                write_json(directory / "eval.json", record)
    return n


def recover_errors(out: Path) -> list[Path]:
    """Archive request directories that must be re-issued.

    Two shapes qualify: a cached result that is an error, and an interrupted
    request (manifest written, no result) left behind by a run that no longer
    holds the lock. Earlier archives (`cN.failedK`) are never touched.
    """
    if lock_is_held(out):
        raise RuntimeError(f"{out / LOCK_NAME} is held: refusing to recover a live run")
    archived: list[Path] = []
    for directory in sorted(out.glob("*/*/*")):
        if not directory.is_dir() or not COPY_DIR_RE.fullmatch(directory.name):
            continue
        result = directory / "result.json"
        if result.exists():
            try:
                if not read_json(result).get("error"):
                    continue
            except (OSError, ValueError):
                continue
        elif not (directory / "request.json").exists():
            continue  # never requested: nothing to recover
        n = 1
        while (target := directory.with_name(f"{directory.name}.failed{n}")).exists():
            n += 1
        directory.rename(target)
        archived.append(target)
    return archived


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
    """One row per (model, problem) from the evaluations on disk."""
    prepared = prepare(cfg, out, tol, strict=False)
    rows: list[dict[str, Any]] = []
    scorers: dict[str, int] = defaultdict(int)
    for model in cfg.models:
        for problem, copies in prepared:
            records = []
            for copy in copies:
                path = job_dir(out, model, problem, copy) / "eval.json"
                if path.exists():
                    records.append(read_json(path))
                    scorers[records[-1].get("scorer", "")] += 1
            if not records:
                continue
            scale = problem.state(tol).scale
            identity = next((r for r in records if r["copy"] == "c0"), None)
            transformed = [r for r in records if r["copy"] != "c0"]
            consistency, n_pairs = pairwise_agreement(records, scale, tol.agreement)
            by_kind: dict[str, list[float]] = defaultdict(list)
            for r in records:
                by_kind[r["kind"]].append(float(r["success"]))
            vs_identity = None
            if identity is not None and identity.get("answered"):
                verdicts = [
                    v
                    for r in transformed
                    if (v := agree(identity, r, scale, tol.agreement)) is not None
                ]
                vs_identity = sum(verdicts) / len(verdicts) if verdicts else None
            success_identity = float(identity["success"]) if identity else None
            success_transformed = (
                sum(r["success"] for r in transformed) / len(transformed)
                if transformed
                else None
            )
            rows.append(
                {
                    "model": model,
                    "problem_id": problem.id,
                    "bucket": problem.bucket,
                    "n_moves": problem.n_moves,
                    "n_expected": len(copies),
                    "n_evaluated": len(records),
                    "answered_rate": sum(bool(r.get("answered")) for r in records)
                    / len(records),
                    "consistency": consistency,
                    "n_pairs": n_pairs,
                    "success_identity": success_identity,
                    "success_transformed": success_transformed,
                    "delta_success": (
                        success_identity - success_transformed
                        if success_identity is not None
                        and success_transformed is not None
                        else None
                    ),
                    "vs_identity": vs_identity,
                    "by_kind": dict(by_kind),
                    "error_kinds": [r.get("error_kind") for r in records],
                }
            )
    return {"config": cfg.name, "rows": rows, "scorers": dict(scorers)}


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else math.nan


def summarize(
    cfg: E0Config, out: Path, tol: ToleranceSpec = TOL
) -> tuple[dict[str, Any], str]:
    data = collect(cfg, out, tol)
    prepared = prepare(cfg, out, tol, strict=False)
    expected_per_model = sum(len(c) for _, c in prepared)
    rng = np.random.default_rng(cfg.seed)
    gate_bucket = f">={E0_GATE.min_moves} moves"
    manifest_path = out / "prepare.json"
    stored = (
        read_json(manifest_path).get("fingerprint", {})
        if manifest_path.exists()
        else {}
    )
    frozen_protocol = stored.get("protocol", "?")
    note = (
        ""
        if frozen_protocol == protocol_hash(cfg)
        else f" (current code: {protocol_hash(cfg)}; read-only)"
    )
    lines = [f"# E0 {cfg.name} — arm {cfg.arm}, set {cfg.problem_set}", ""]
    lines.append(f"Frozen inputs: protocol {frozen_protocol}{note}")
    lines.append("")
    lines.append(
        "| model | bucket | problems | coverage | answered | success (identity) | "
        "success (transformed) | Δsuccess = identity − transformed [CI95] | "
        "consistency [CI95] | vs identity |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    summary: list[dict[str, Any]] = []
    gate_rows: dict[str, dict[str, Any]] = {}
    for model in cfg.models:
        rows = [r for r in data["rows"] if r["model"] == model]
        buckets = [
            n for n, _, _ in LENGTH_BUCKETS if any(r["bucket"] == n for r in rows)
        ]
        for bucket in buckets + ["all", gate_bucket]:
            if bucket == "all":
                sel = rows
                expected = expected_per_model
            elif bucket == gate_bucket:
                sel = [r for r in rows if r["n_moves"] >= E0_GATE.min_moves]
                expected = sum(
                    len(c) for p, c in prepared if p.n_moves >= E0_GATE.min_moves
                )
            else:
                sel = [r for r in rows if r["bucket"] == bucket]
                expected = sum(len(c) for p, c in prepared if p.bucket == bucket)
            if not sel:
                continue
            cons = [r["consistency"] for r in sel if r["consistency"] is not None]
            deltas = [r["delta_success"] for r in sel if r["delta_success"] is not None]
            vs_id = [r["vs_identity"] for r in sel if r["vs_identity"] is not None]
            evaluated = sum(r["n_evaluated"] for r in sel)
            entry = {
                "model": model,
                "bucket": bucket,
                "n_problems": len(sel),
                "coverage": evaluated / expected if expected else math.nan,
                "n_missing": expected - evaluated,
                "answered": _mean([r["answered_rate"] for r in sel]),
                "success_identity": _mean(
                    [
                        r["success_identity"]
                        for r in sel
                        if r["success_identity"] is not None
                    ]
                ),
                "success_transformed": _mean(
                    [
                        r["success_transformed"]
                        for r in sel
                        if r["success_transformed"] is not None
                    ]
                ),
                "delta_success": _mean(deltas),
                "delta_success_ci": list(_bootstrap_ci(deltas, rng, cfg.bootstrap)),
                "consistency": _mean(cons),
                "consistency_ci": list(_bootstrap_ci(cons, rng, cfg.bootstrap)),
                "n_consistency": len(cons),
                "vs_identity": _mean(vs_id),
            }
            summary.append(entry)
            if bucket == gate_bucket:
                gate_rows[model] = entry
            dlo, dhi = entry["delta_success_ci"]
            clo, chi = entry["consistency_ci"]
            lines.append(
                f"| {model} | {bucket} | {entry['n_problems']} | {entry['coverage']:.3f} | "
                f"{entry['answered']:.3f} | {entry['success_identity']:.3f} | "
                f"{entry['success_transformed']:.3f} | {entry['delta_success']:+.3f} "
                f"[{dlo:+.3f}, {dhi:+.3f}] | {entry['consistency']:.3f} "
                f"[{clo:.3f}, {chi:.3f}] (n={entry['n_consistency']}) | "
                f"{entry['vs_identity']:.3f} |"
            )
    # success by transform kind
    lines.append("")
    lines.append("| model | kind | copies | success |")
    lines.append("|---|---|---|---|")
    by_kind: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in data["rows"]:
        for kind, values in r["by_kind"].items():
            by_kind[(r["model"], kind)].extend(values)
    for (model, kind), vals in sorted(by_kind.items()):
        lines.append(f"| {model} | {kind} | {len(vals)} | {_mean(vals):.3f} |")
    # kill gate (AC-13 / AC-14)
    lines.append("")
    gate: dict[str, Any] = {"gated": cfg.gated, "verdict": None, "models": {}}
    if not cfg.gated:
        gated = ", ".join(c.name for c in experiment_configs if c.gated)
        lines.append(f"Not a gated config: the H0 verdict is read from {gated}.")
    else:
        falsified = False
        for model in cfg.models:
            entry = gate_rows.get(model)
            coverage_reasons: list[str] = []
            quality_reasons: list[str] = []
            if entry is None:
                coverage_reasons.append("no evaluations in the gate bucket")
            else:
                if entry["n_missing"] > 0:
                    coverage_reasons.append(
                        f"{entry['n_missing']} copies not evaluated"
                    )
                if entry["answered"] < E0_GATE.min_answered_rate:
                    quality_reasons.append(
                        f"answered {entry['answered']:.2f} < "
                        f"{E0_GATE.min_answered_rate:.2f}"
                    )
                if entry["success_identity"] < E0_GATE.min_success_identity:
                    quality_reasons.append(
                        f"success(identity) {entry['success_identity']:.2f} < "
                        f"{E0_GATE.min_success_identity:.2f}"
                    )
            reasons = coverage_reasons + quality_reasons
            status = {
                "consistency": entry["consistency"] if entry else None,
                "applicable": not reasons,
                "incomplete": bool(coverage_reasons),
                "withheld": reasons,
            }
            gate["models"][model] = status
            if reasons:
                lines.append(
                    f"Gate ({gate_bucket}) {model}: verdict withheld — "
                    + "; ".join(reasons)
                )
            else:
                hit = entry["consistency"] >= E0_GATE.consistency_kill
                falsified = falsified or hit
                lines.append(
                    f"Gate ({gate_bucket}) {model}: consistency {entry['consistency']:.3f} "
                    f"vs kill {E0_GATE.consistency_kill:.2f} -> "
                    + ("at/above the gate" if hit else "below the gate")
                )
        applicable = [m for m, st in gate["models"].items() if st["applicable"]]
        incomplete = [m for m, st in gate["models"].items() if st["incomplete"]]
        excluded = [
            m
            for m, st in gate["models"].items()
            if not st["applicable"] and not st["incomplete"]
        ]
        if falsified:
            # One consistent model is decisive regardless of the others.
            gate["verdict"] = "H0 FALSIFIED"
            lines.append(
                "Gate verdict: H0 FALSIFIED — an open model is consistent at/above "
                "the kill gate."
            )
        elif incomplete:
            # The best model may be among the incomplete ones: no verdict yet.
            gate["verdict"] = "withheld"
            lines.append(
                "Gate verdict: withheld — incomplete data for "
                + ", ".join(incomplete)
                + "."
            )
        elif not applicable:
            gate["verdict"] = "withheld"
            lines.append(
                "Gate verdict: withheld — no model meets the answered/success "
                "conditions (" + ", ".join(excluded) + ")."
            )
        else:
            gate["verdict"] = "H0 stands"
            note = (
                " Excluded for answered/success conditions: "
                + ", ".join(excluded)
                + "."
                if excluded
                else ""
            )
            lines.append(
                "Gate verdict: H0 stands — every applicable model ("
                + ", ".join(applicable)
                + ") is below the gate."
                + note
            )
    evaluated_models = {r["model"] for r in data["rows"]}
    unevaluated = [m for m in cfg.models if m not in evaluated_models]
    if unevaluated:
        lines.append("")
        lines.append("No evaluations for: " + ", ".join(unevaluated) + ".")
    scorers = data.get("scorers", {})
    if len(scorers) > 1 or (scorers and dsl.scorer_hash() not in scorers):
        lines.append("")
        lines.append(
            "WARNING: records were scored by different parser versions ("
            + ", ".join(
                f"{k or 'unversioned'}: {v}" for k, v in sorted(scorers.items())
            )
            + f"); current is {dsl.scorer_hash()}. Run --rescore before reading."
        )
    text = "\n".join(lines) + "\n"
    payload = {
        "config": cfg.name,
        "summary": summary,
        "gate": gate,
        "unevaluated_models": unevaluated,
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
    ap.add_argument(
        "--recover-errors",
        action="store_true",
        help="archive request directories whose result is an error",
    )
    ap.add_argument(
        "--rescore",
        action="store_true",
        help="re-evaluate cached model outputs with the current scorer (offline)",
    )
    ap.add_argument(
        "--force", action="store_true", help="rescore even while a run is live"
    )
    ap.add_argument(
        "--limit", type=int, default=None, help="only N problems, taken across buckets"
    )
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
            f"[{cfg.name}] {len(prepared)} problems ({dict(buckets)}), {n_copies} copies "
            f"({dict(kinds)}), {n_copies * len(cfg.models)} requests over "
            f"{len(cfg.models)} model(s); out={out}"
        )
    if args.recover_errors:
        archived = recover_errors(out)
        print(f"[{cfg.name}] archived {len(archived)} failed request directories")
    if args.rescore:
        changed = rescore(cfg, out, force=args.force)
        print(f"[{cfg.name}] rescored cached outputs; {changed} evaluations changed")
    if args.run:
        code = run(cfg, out, limit=args.limit, workers=args.workers)
    if args.summary:
        payload, text = summarize(cfg, out)
        write_json(out / "summary.json", payload)
        (out / "summary.md").write_text(text)
        sys.stdout.write(text)
    if not (
        args.dry or args.run or args.summary or args.recover_errors or args.rescore
    ):
        ap.print_help()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
