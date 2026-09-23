"""Construction problems and their trajectories, from a seed.

A `Problem` is a construction problem in the glossary's sense: given points,
a target expression (a nested macro over them), and the trajectory of
primitive moves that produces the target. The target object is whatever the
expansion produced — there is no second implementation to disagree with it —
and the goal predicate is coincidence with that object. Problems are grouped
into length buckets by the number of moves in their trajectory, after every
move the target does not depend on has been pruned.

Everything is deterministic in the seed (AC-9): the same seed gives
byte-identical problem files, and the generator reports how many candidates
it rejected and why.

    python -m experiments.geostate.generator --seed 1 --per-bucket 5 --show
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import random
from typing import Any

from .configs import (
    GENERATOR,
    LENGTH_BUCKETS,
    TOL,
    GeneratorSpec,
    ToleranceSpec,
    bucket_of,
)
from .macros import (
    MACROS,
    POINT_MACROS,
    Builder,
    Expr,
    describe,
    expand,
    validate_expr,
)
from .state import (
    ERR_DEGENERATE,
    POINT,
    GeometricState,
    Move,
    MoveError,
    same_geometry,
)

Given = dict[str, tuple[float, float]]


@dataclass
class Problem:
    id: str
    seed: int
    given: Given
    expr: Expr
    description: str
    target_kind: str
    target_params: tuple[float, ...]
    trajectory: list[Move]
    answer: str
    n_moves: int
    bucket: str
    source: str = "synthetic"
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def labels(self) -> list[str]:
        return list(self.given)

    def state(self, tol: ToleranceSpec = TOL) -> GeometricState:
        return GeometricState.from_given(self.given, tol)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "seed": self.seed,
            "given": {k: list(v) for k, v in self.given.items()},
            "expr": self.expr,
            "description": self.description,
            "target_kind": self.target_kind,
            "target_params": list(self.target_params),
            "trajectory": [m.to_dict() for m in self.trajectory],
            "answer": self.answer,
            "n_moves": self.n_moves,
            "bucket": self.bucket,
            "source": self.source,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Problem:
        return cls(
            id=d["id"],
            seed=int(d["seed"]),
            given={k: (float(v[0]), float(v[1])) for k, v in d["given"].items()},
            expr=d["expr"],
            description=d["description"],
            target_kind=d["target_kind"],
            target_params=tuple(float(x) for x in d["target_params"]),
            trajectory=[Move.from_dict(m) for m in d["trajectory"]],
            answer=d["answer"],
            n_moves=int(d["n_moves"]),
            bucket=d["bucket"],
            source=d.get("source", "synthetic"),
            meta=dict(d.get("meta", {})),
        )


def prune_trajectory(
    given: Given, moves: list[Move], answer: str, tol: ToleranceSpec = TOL
) -> tuple[list[Move], str]:
    """Drop every move the answer does not depend on; renumber the handles.

    Moves keep their relative order, so the answer's producing move is the
    last one of the pruned trajectory.
    """
    state = GeometricState.from_given(given, tol)
    produced_by: dict[str, int] = {}
    outputs: list[list[str]] = []
    for i, move in enumerate(moves):
        objs = state.apply(move)
        outputs.append([o.handle for o in objs])
        for o in objs:
            produced_by[o.handle] = i
    needed: set[int] = set()
    stack = [answer]
    while stack:
        handle = stack.pop()
        if handle in produced_by and produced_by[handle] not in needed:
            index = produced_by[handle]
            needed.add(index)
            stack.extend(moves[index].args)
    keep = sorted(needed)
    if len(keep) == len(moves):
        return list(moves), answer
    replay = GeometricState.from_given(given, tol)
    mapping = {label: label for label in given}
    pruned: list[Move] = []
    for index in keep:
        move = Move(moves[index].op, tuple(mapping[a] for a in moves[index].args))
        objs = replay.apply(move)
        pruned.append(move)
        for old, new in zip(outputs[index], objs, strict=True):
            mapping[old] = new.handle
    return pruned, mapping[answer]


def build_problem(
    problem_id: str,
    seed: int,
    given: Given,
    expr: Expr,
    tol: ToleranceSpec = TOL,
    source: str = "synthetic",
    meta: dict[str, Any] | None = None,
    max_extent: float | None = None,
) -> Problem:
    """Expand `expr` on the given points; raises MoveError if degenerate.

    Guarantees AC-7: the target is produced by the trajectory's last move and
    no object that exists before that move coincides with it, so the goal
    predicate is false on every earlier state.
    """
    validate_expr(expr)
    scratch = GeometricState.from_given(given, tol)
    builder = Builder(scratch)
    handle = expand(builder, expr)
    if not builder.moves:
        raise MoveError(ERR_DEGENERATE, "target is a given point")
    if handle in given:
        raise MoveError(ERR_DEGENERATE, "target is a given point")
    moves, answer = prune_trajectory(given, builder.moves, handle, tol)
    state = GeometricState.from_given(given, tol)
    state.replay(moves)
    target = state.get(answer)
    last = len(moves) - 1
    if target.move_index != last:
        raise MoveError(ERR_DEGENERATE, "target existed before the last move")
    for o in state.objects:
        if o.move_index < last and same_geometry(
            o, target, state.scale, tol.coincidence
        ):
            raise MoveError(ERR_DEGENERATE, "target existed before the last move")
    if max_extent is not None:
        cx = sum(x for x, _ in given.values()) / len(given)
        cy = sum(y for _, y in given.values()) / len(given)
        limit = max_extent * state.scale
        for o in state.objects:
            if o.kind == POINT and math.dist(o.xy(), (cx, cy)) > limit:
                raise MoveError(ERR_DEGENERATE, "construction leaves the working area")
    return Problem(
        id=problem_id,
        seed=seed,
        given=dict(given),
        expr=expr,
        description=describe(expr),
        target_kind=target.kind,
        target_params=tuple(target.params()),
        trajectory=moves,
        answer=answer,
        n_moves=len(moves),
        bucket=bucket_of(len(moves)),
        source=source,
        meta=dict(meta or {}),
    )


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample_given(rng: random.Random, spec: GeneratorSpec, n: int) -> Given:
    """n distinct integer points on the grid, at least 2 units apart."""
    labels = spec.labels[:n]
    pts: list[tuple[float, float]] = []
    tries = 0
    while len(pts) < n:
        tries += 1
        if tries > 1000:
            raise MoveError(ERR_DEGENERATE, "could not place given points")
        x = float(rng.randint(spec.coord_min, spec.coord_max))
        y = float(rng.randint(spec.coord_min, spec.coord_max))
        if all(math.hypot(x - px, y - py) >= 2.0 for px, py in pts):
            pts.append((x, y))
    return dict(zip(labels, pts, strict=True))


def count_macros(expr: Expr) -> int:
    if isinstance(expr, str):
        return 0
    return 1 + sum(count_macros(a) for a in expr[1:])


def sample_expr(
    rng: random.Random,
    spec: GeneratorSpec,
    labels: list[str],
    depth: int = 0,
    budget: list[int] | None = None,
) -> Expr:
    """A random expression with at most `spec.max_macros` macro nodes."""
    budget = [spec.max_macros] if budget is None else budget
    names = list(MACROS) if depth == 0 else list(POINT_MACROS)
    names = [n for n in names if MACROS[n].arity <= len(labels)]
    name = rng.choice(names)
    macro = MACROS[name]
    budget[0] -= 1
    args: list[Expr] = rng.sample(labels, macro.arity)
    for i in range(macro.arity):
        if (
            depth < spec.max_depth
            and budget[0] > 0
            and rng.random() < spec.compose_prob
        ):
            args[i] = sample_expr(rng, spec, labels, depth + 1, budget)
    return [name, *args]


@dataclass
class GenerationStats:
    attempts: int = 0
    accepted: int = 0
    rejected: dict[str, int] = field(default_factory=dict)
    underfilled: dict[str, int] = field(default_factory=dict)  # bucket -> shortfall

    def reject(self, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempts": self.attempts,
            "accepted": self.accepted,
            "rejected": dict(sorted(self.rejected.items(), key=lambda kv: -kv[1])),
            "underfilled": self.underfilled,
        }


def _reason(exc: Exception) -> str:
    text = str(exc)
    return text.split(": ", 1)[1] if ": " in text else text


def generate_with_stats(
    spec: GeneratorSpec = GENERATOR,
    tol: ToleranceSpec = TOL,
    seed: int = 0,
    per_bucket: int = 10,
    prefix: str = "syn",
) -> tuple[list[Problem], GenerationStats]:
    """Fill every length bucket with `per_bucket` problems, deterministically."""
    rng = random.Random(seed)
    stats = GenerationStats()
    buckets: dict[str, list[Problem]] = {name: [] for name, _, _ in LENGTH_BUCKETS}
    budget = per_bucket * len(buckets) * spec.max_attempts_per_problem
    while stats.attempts < budget and any(
        len(b) < per_bucket for b in buckets.values()
    ):
        stats.attempts += 1
        n = rng.randint(spec.n_given_min, spec.n_given_max)
        try:
            given = sample_given(rng, spec, n)
            expr = sample_expr(rng, spec, list(given))
            problem = build_problem(
                f"{prefix}-{seed}-{stats.attempts:06d}",
                seed,
                given,
                expr,
                tol,
                max_extent=tol.max_extent,
            )
        except (MoveError, ValueError) as exc:
            stats.reject(_reason(exc))
            continue
        if problem.n_moves > spec.max_moves:
            stats.reject("too many moves")
            continue
        bucket = buckets[problem.bucket]
        if len(bucket) >= per_bucket:
            stats.reject(f"bucket {problem.bucket} full")
            continue
        bucket.append(problem)
        stats.accepted += 1
    for name, problems in buckets.items():
        if len(problems) < per_bucket:
            stats.underfilled[name] = per_bucket - len(problems)
    out: list[Problem] = []
    for name, _, _ in LENGTH_BUCKETS:
        out.extend(buckets[name])
    return out, stats


def generate(
    spec: GeneratorSpec = GENERATOR,
    tol: ToleranceSpec = TOL,
    seed: int = 0,
    per_bucket: int = 10,
    prefix: str = "syn",
) -> list[Problem]:
    problems, stats = generate_with_stats(spec, tol, seed, per_bucket, prefix)
    if stats.underfilled:
        raise RuntimeError(
            f"generator could not fill buckets {stats.underfilled} within "
            f"{stats.attempts} attempts; rejections: {stats.to_dict()['rejected']}"
        )
    return problems


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def save_problems(problems: list[Problem], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as fh:
        for p in problems:
            fh.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")
    tmp.replace(path)
    return path


def load_problems(path: str | Path) -> list[Problem]:
    out = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            out.append(Problem.from_dict(json.loads(line)))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-bucket", type=int, default=5)
    ap.add_argument("--out", type=Path, default=None, help="write problems.jsonl here")
    ap.add_argument(
        "--show", action="store_true", help="print each problem's trajectory"
    )
    args = ap.parse_args(argv)
    problems, stats = generate_with_stats(seed=args.seed, per_bucket=args.per_bucket)
    counts: dict[str, int] = {}
    for p in problems:
        counts[p.bucket] = counts.get(p.bucket, 0) + 1
    print(f"{len(problems)} problems; per bucket: {counts}")
    print(f"attempts: {stats.attempts}, accepted: {stats.accepted}")
    for reason, n in stats.to_dict()["rejected"].items():
        print(f"  rejected {n:5d}  {reason}")
    if stats.underfilled:
        print(f"UNDERFILLED: {stats.underfilled}")
    if args.show:
        from .dsl import trajectory_program

        for p in problems:
            print(f"\n[{p.id}] {p.bucket} ({p.n_moves} moves) given {p.given}")
            print(f"Construct {p.description}.")
            print(trajectory_program(p.trajectory, p.answer, p.state()))
    if args.out is not None:
        print(f"wrote {save_problems(problems, args.out)}")
    return 1 if stats.underfilled else 0


if __name__ == "__main__":
    raise SystemExit(main())
