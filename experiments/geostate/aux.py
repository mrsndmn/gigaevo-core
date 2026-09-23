"""The auxiliary construction set: NoReGeo point stems recast as constructions.

Seven NoReGeo stems ask for a point that a straightedge-and-compass
construction produces (midpoint, line intersection, foot of the angle
bisector, incentre, circumcentre, reflection, rotation). Each row becomes a
`Problem` whose goal is coincidence with the constructed point; the row's
golden answer is used only to *validate the kernel* (AC-8) and never enters a
prompt. The benchmark's numeric and classification stems are out of scope.

Reflection across a coordinate axis and rotation about the origin need objects
that are not in the row's `points`: the axis is given as two points on it and
the centre as a point, so that every problem is "labelled points -> target"
and a similarity transform moves the axis with the rest of the figure.

    python -m experiments.geostate.aux --check      # AC-8 over all 700 rows
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import random
import re
from typing import Any

from experiments.noregeo_synth import configs as _noregeo_configs
from experiments.noregeo_synth import dataset as _dataset

from .configs import AUX_STEMS, TOL, ToleranceSpec
from .generator import Given, Problem, build_problem
from .macros import Expr
from .state import MoveError

# The point tolerance NoReGeo traces are scored with; the kernel must
# reproduce the golden point within it.
POINT_ATOL: float = float(_noregeo_configs.EXPERIMENT_CONFIGS[0].point_atol)

AXIS_RE = re.compile(r"across the ([XY])-axis")
ANGLE_RE = re.compile(r"rotated by (\d+)°")
VERTEX_RE = re.compile(r"bisector at ([A-Z])")

ROTATIONS = {90: "rotate90", 180: "rotate180", 270: "rotate270"}


@dataclass(frozen=True)
class Skipped:
    task_id: str
    reason: str


def _points(row: dict[str, Any]) -> Given:
    pts = row["points"]
    if isinstance(pts, dict):
        return {str(k): (float(v[0]), float(v[1])) for k, v in pts.items()}
    labels = "ABCDEFGH"
    return {labels[i]: (float(p[0]), float(p[1])) for i, p in enumerate(pts)}


def _expr_and_given(stem: str, row: dict[str, Any]) -> tuple[Expr, Given]:
    given = _points(row)
    macro = AUX_STEMS[stem]
    labels = list(given)
    if macro == "midpoint":
        return ["midpoint", labels[0], labels[1]], given
    if macro == "line_meet":
        return ["line_meet", *labels[:4]], given
    if macro in ("incenter", "circumcenter"):
        return [macro, *labels[:3]], given
    if macro == "angle_bisector_foot":
        m = VERTEX_RE.search(row["question"])
        if not m or m.group(1) not in given:
            raise ValueError("no vertex named in the question")
        vertex = m.group(1)
        others = [x for x in labels[:3] if x != vertex]
        return ["angle_bisector_foot", vertex, others[0], others[1]], given
    if macro == "reflect":
        m = AXIS_RE.search(row["question"])
        if not m:
            raise ValueError("no axis named in the question")
        p = labels[0]
        given = dict(given)
        given["O"] = (0.0, 0.0)
        given["U"] = (1.0, 0.0) if m.group(1) == "X" else (0.0, 1.0)
        return ["reflect", p, "O", "U"], given
    if macro == "rotate":
        m = ANGLE_RE.search(row["question"])
        if not m or int(m.group(1)) not in ROTATIONS:
            raise ValueError("unsupported rotation angle")
        p = labels[0]
        given = dict(given)
        given["O"] = (0.0, 0.0)
        return [ROTATIONS[int(m.group(1))], p, "O"], given
    raise ValueError(f"no recipe for stem {stem}")


def problem_from_task(task: _dataset.Task, tol: ToleranceSpec = TOL) -> Problem:
    """One NoReGeo row -> construction problem, validated against its golden point."""
    expr, given = _expr_and_given(task.stem, task.row)
    problem = build_problem(
        f"aux-{task.stem}-{task.qid}",
        seed=0,
        given=given,
        expr=expr,
        tol=tol,
        source=f"aux:{task.stem}:{task.qid}",
        meta={"stem": task.stem, "qid": task.qid},
    )
    gold = task.row["answer"]
    gx, gy = (gold["point"] if isinstance(gold, dict) else gold)[:2]
    dist = math.hypot(
        problem.target_params[0] - float(gx), problem.target_params[1] - float(gy)
    )
    if dist > POINT_ATOL:
        raise ValueError(
            f"kernel target {problem.target_params} vs golden ({gx}, {gy}): |d|={dist:.4g}"
        )
    return problem


def aux_tasks() -> list[_dataset.Task]:
    return [t for t in _dataset.load_tasks() if t.stem in AUX_STEMS]


def check_all(tol: ToleranceSpec = TOL) -> tuple[list[Problem], list[Skipped]]:
    """Recast every row of the seven stems; return the problems and the skips."""
    ok: list[Problem] = []
    skipped: list[Skipped] = []
    for task in aux_tasks():
        try:
            ok.append(problem_from_task(task, tol))
        except (MoveError, ValueError) as exc:
            skipped.append(Skipped(task.task_id, str(exc)))
    return ok, skipped


def load_aux_problems(
    per_stem: int, seed: int, tol: ToleranceSpec = TOL
) -> tuple[list[Problem], list[Skipped]]:
    """A deterministic per-stem sample of valid auxiliary problems."""
    rng = random.Random(seed)
    by_stem: dict[str, list[_dataset.Task]] = {}
    for task in aux_tasks():
        by_stem.setdefault(task.stem, []).append(task)
    problems: list[Problem] = []
    skipped: list[Skipped] = []
    for stem in AUX_STEMS:
        tasks = sorted(by_stem.get(stem, []), key=lambda t: t.qid)
        rng.shuffle(tasks)
        taken = 0
        for task in tasks:
            if taken >= per_stem:
                break
            try:
                problems.append(problem_from_task(task, tol))
                taken += 1
            except (MoveError, ValueError) as exc:
                skipped.append(Skipped(task.task_id, str(exc)))
    return problems, skipped


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check", action="store_true", help="validate all rows against golden"
    )
    args = ap.parse_args(argv)
    if args.check:
        ok, skipped = check_all()
        per_stem: dict[str, int] = {}
        for p in ok:
            per_stem[p.meta["stem"]] = per_stem.get(p.meta["stem"], 0) + 1
        print(f"valid: {len(ok)} rows; per stem: {per_stem}")
        moves: dict[str, list[int]] = {}
        for p in ok:
            moves.setdefault(p.meta["stem"], []).append(p.n_moves)
        for stem, ns in moves.items():
            print(f"  {stem}: moves {min(ns)}-{max(ns)}")
        print(f"skipped: {len(skipped)}")
        for s in skipped[:50]:
            print(f"  {s.task_id}: {s.reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
