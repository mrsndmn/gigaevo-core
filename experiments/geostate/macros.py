"""Construction targets as macros over the three primitives.

A macro names a classical construction (midpoint, circumcentre, reflection,
...) and knows how to *expand* it into primitive moves on a `GeometricState`.
The expansion is the only implementation: the target object of a construction
problem is whatever the expansion produces, and the recorded trajectory is the
sequence of moves it took. Macros are never actions in the probe (ADR 0005
context; Round 5 of the interview): the model only ever sees `line`, `circle`
and `meet`.

A target *expression* is a nested list: `["midpoint", "A", ["circumcenter",
"A", "B", "C"]]`. Point-valued sub-expressions may fill any point slot; the
phrase templates are written so that nesting reads naturally ("the midpoint of
A and the circumcentre of the triangle with vertices A, B and C").

`Builder` wraps a state and reuses an object that already exists with the same
geometry instead of constructing it twice; `generator.prune_trajectory` then
drops every move the target does not depend on, so a recorded trajectory
carries no dead moves.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math

from . import cga
from .state import (
    CIRCLE,
    ERR_DEGENERATE,
    LINE,
    OP_CIRCLE,
    OP_LINE,
    OP_MEET,
    POINT,
    GeometricState,
    Move,
    MoveError,
    Obj,
    incident,
)

Expr = str | list


class Builder:
    """Applies moves to a state, reusing existing geometry, recording the trajectory."""

    def __init__(self, state: GeometricState):
        self.state = state
        self.moves: list[Move] = []

    def _find_or_apply(self, move: Move) -> list[str]:
        produced = self.state.compute(move)
        found = [self.state.find(kind, mv) for kind, mv in produced]
        if all(f is not None for f in found):
            return [f.handle for f in found]  # type: ignore[union-attr]
        objs = self.state.apply(move)
        self.moves.append(move)
        return [o.handle for o in objs]

    def line(self, p: str, q: str) -> str:
        return self._find_or_apply(Move(OP_LINE, (p, q)))[0]

    def circle(self, centre: str, through: str) -> str:
        return self._find_or_apply(Move(OP_CIRCLE, (centre, through)))[0]

    def meet(self, a: str, b: str) -> list[str]:
        return self._find_or_apply(Move(OP_MEET, (a, b)))

    # -- selection helpers (all invariant under similarities) ---------------

    def xy(self, handle: str) -> tuple[float, float]:
        return self.state.get(handle).xy()

    def other(self, candidates: Sequence[str], not_this: str) -> str:
        """The candidate that does not coincide with `not_this`."""
        others = [h for h in candidates if not self.state.coincides(h, not_this)]
        if len(others) != 1:
            raise MoveError(ERR_DEGENERATE, "expected exactly one new point")
        return others[0]

    def on_ray(self, candidates: Sequence[str], origin: str, towards: str) -> str:
        """The candidate on the ray origin -> towards (dot product > 0)."""
        ox, oy = self.xy(origin)
        tx, ty = self.xy(towards)
        for h in candidates:
            x, y = self.xy(h)
            if (x - ox) * (tx - ox) + (y - oy) * (ty - oy) > 0:
                return h
        raise MoveError(ERR_DEGENERATE, "no candidate on the ray")

    def ccw(
        self, candidates: Sequence[str], origin: str, from_pt: str, want_ccw: bool
    ) -> str:
        """The candidate at +90 (ccw) or -90 (cw) from `from_pt` about `origin`."""
        ox, oy = self.xy(origin)
        fx, fy = self.xy(from_pt)
        for h in candidates:
            x, y = self.xy(h)
            cross = (fx - ox) * (y - oy) - (fy - oy) * (x - ox)
            if (cross > 0) == want_ccw:
                return h
        raise MoveError(ERR_DEGENERATE, "no candidate with the requested orientation")

    def point_on_line(self, p: str, a: str, q: str) -> bool:
        """Is p on the line through a and q? Computed, not constructed."""
        kind, mv = self.state.compute(Move(OP_LINE, (a, q)))[0]
        return incident(
            self.state.get(p),
            Obj("?", kind, mv),
            self.state.scale,
            self.state.tol.incidence,
        )


# ---------------------------------------------------------------------------
# Expansions (Euclidea-style, primitives only)
# ---------------------------------------------------------------------------


def perp_bisector(b: Builder, p: str, q: str) -> str:
    c1 = b.circle(p, q)
    c2 = b.circle(q, p)
    x, y = b.meet(c1, c2)
    return b.line(x, y)


def midpoint(b: Builder, p: str, q: str) -> str:
    bis = perp_bisector(b, p, q)
    return b.meet(bis, b.line(p, q))[0]


def circumcenter(b: Builder, p: str, q: str, r: str) -> str:
    return b.meet(perp_bisector(b, p, q), perp_bisector(b, q, r))[0]


def circumcircle(b: Builder, p: str, q: str, r: str) -> str:
    return b.circle(circumcenter(b, p, q, r), p)


def reflect(b: Builder, p: str, a: str, q: str) -> str:
    """Reflection of p across the line through a and q."""
    x, y = b.meet(b.circle(a, p), b.circle(q, p))
    return b.other([x, y], p)


def foot(b: Builder, p: str, a: str, q: str) -> str:
    """Foot of the perpendicular from p to the line through a and q."""
    pr = reflect(b, p, a, q)
    return b.meet(b.line(p, pr), b.line(a, q))[0]


def perp_through(b: Builder, p: str, a: str, q: str) -> str:
    """Line through p perpendicular to the line through a and q."""
    if b.point_on_line(p, a, q):
        base = b.line(a, q)
        anchor = q if b.state.coincides(p, a) else a
        x, y = b.meet(b.circle(p, anchor), base)
        return perp_bisector(b, x, y)
    return b.line(p, reflect(b, p, a, q))


def parallel_through(b: Builder, p: str, a: str, q: str) -> str:
    """Line through p parallel to the line through a and q (rhombus construction)."""
    if b.point_on_line(p, a, q):
        raise MoveError(
            ERR_DEGENERATE, "point lies on the line: parallel is the line itself"
        )
    base = b.line(a, q)
    x, _ = b.meet(b.circle(a, p), base)
    u, v = b.meet(b.circle(p, a), b.circle(x, a))
    r = b.other([u, v], a)
    return b.line(p, r)


def angle_bisector(b: Builder, a: str, p: str, q: str) -> str:
    """Bisector of the angle at a formed by p and q."""
    x, y = b.meet(b.circle(a, p), b.line(a, q))
    d = b.on_ray([x, y], a, q)
    return perp_bisector(b, p, d)


def angle_bisector_foot(b: Builder, a: str, p: str, q: str) -> str:
    return b.meet(angle_bisector(b, a, p, q), b.line(p, q))[0]


def incenter(b: Builder, a: str, p: str, q: str) -> str:
    return b.meet(angle_bisector(b, a, p, q), angle_bisector(b, p, a, q))[0]


def line_meet(b: Builder, a: str, p: str, c: str, d: str) -> str:
    return b.meet(b.line(a, p), b.line(c, d))[0]


def rotate180(b: Builder, p: str, o: str) -> str:
    x, y = b.meet(b.circle(o, p), b.line(o, p))
    return b.other([x, y], p)


def _rotate_quarter(b: Builder, p: str, o: str, ccw: bool) -> str:
    pp = rotate180(b, p, o)
    perp = perp_bisector(b, p, pp)
    x, y = b.meet(perp, b.circle(o, p))
    return b.ccw([x, y], o, p, ccw)


def rotate90(b: Builder, p: str, o: str) -> str:
    return _rotate_quarter(b, p, o, True)


def rotate270(b: Builder, p: str, o: str) -> str:
    return _rotate_quarter(b, p, o, False)


def centroid(b: Builder, p: str, q: str, r: str) -> str:
    m1 = midpoint(b, p, q)
    m2 = midpoint(b, q, r)
    return b.meet(b.line(r, m1), b.line(p, m2))[0]


@dataclass(frozen=True)
class Macro:
    name: str
    arity: int
    result: str
    template: str
    fn: Callable[..., str]


MACROS: dict[str, Macro] = {
    m.name: m
    for m in (
        Macro("midpoint", 2, POINT, "the midpoint of {0} and {1}", midpoint),
        Macro(
            "perp_bisector",
            2,
            LINE,
            "the perpendicular bisector of the segment from {0} to {1}",
            perp_bisector,
        ),
        Macro(
            "circumcenter",
            3,
            POINT,
            "the circumcentre of the triangle with vertices {0}, {1} and {2}",
            circumcenter,
        ),
        Macro(
            "circumcircle",
            3,
            CIRCLE,
            "the circle through {0}, {1} and {2}",
            circumcircle,
        ),
        Macro(
            "incenter",
            3,
            POINT,
            "the incentre of the triangle with vertices {0}, {1} and {2}",
            incenter,
        ),
        Macro(
            "centroid",
            3,
            POINT,
            "the centroid of the triangle with vertices {0}, {1} and {2}",
            centroid,
        ),
        Macro(
            "reflect",
            3,
            POINT,
            "the reflection of {0} across the line through {1} and {2}",
            reflect,
        ),
        Macro(
            "foot",
            3,
            POINT,
            "the foot of the perpendicular from {0} to the line through {1} and {2}",
            foot,
        ),
        Macro(
            "perp_through",
            3,
            LINE,
            "the line through {0} perpendicular to the line through {1} and {2}",
            perp_through,
        ),
        Macro(
            "parallel_through",
            3,
            LINE,
            "the line through {0} parallel to the line through {1} and {2}",
            parallel_through,
        ),
        Macro(
            "angle_bisector",
            3,
            LINE,
            "the bisector of the angle at {0} formed by {1} and {2}",
            angle_bisector,
        ),
        Macro(
            "angle_bisector_foot",
            3,
            POINT,
            "the point where the bisector of the angle at {0} formed by {1} and {2} "
            "meets the line through {1} and {2}",
            angle_bisector_foot,
        ),
        Macro(
            "line_meet",
            4,
            POINT,
            "the intersection of the line through {0} and {1} with the line "
            "through {2} and {3}",
            line_meet,
        ),
        Macro(
            "rotate90",
            2,
            POINT,
            "the image of {0} under a 90-degree counter-clockwise rotation about {1}",
            rotate90,
        ),
        Macro(
            "rotate270",
            2,
            POINT,
            "the image of {0} under a 90-degree clockwise rotation about {1}",
            rotate270,
        ),
        Macro(
            "rotate180",
            2,
            POINT,
            "the point symmetric to {0} with respect to {1}",
            rotate180,
        ),
    )
}

POINT_MACROS = tuple(m.name for m in MACROS.values() if m.result == POINT)


# ---------------------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------------------


def validate_expr(expr: Expr) -> None:
    if isinstance(expr, str):
        return
    if not isinstance(expr, list) or not expr or not isinstance(expr[0], str):
        raise ValueError(f"bad expression {expr!r}")
    macro = MACROS.get(expr[0])
    if macro is None:
        raise ValueError(f"unknown macro {expr[0]!r}")
    args = expr[1:]
    if len(args) != macro.arity:
        raise ValueError(f"{macro.name} takes {macro.arity} arguments, got {len(args)}")
    for a in args:
        validate_expr(a)
        if isinstance(a, list) and MACROS[a[0]].result != POINT:
            raise ValueError("only point-valued macros may be nested")


def expand(builder: Builder, expr: Expr, strict: bool = False) -> str:
    """Expand an expression on the builder's state; returns the result handle.

    With `strict`, every macro node must add at least one move: a node whose
    result already existed (a construction that restates a sub-result) is
    rejected as degenerate, so descriptions never describe nothing.
    """
    if isinstance(expr, str):
        builder.state.get(expr)
        return expr
    macro = MACROS[expr[0]]
    handles = [expand(builder, a, strict) for a in expr[1:]]
    before = len(builder.moves)
    handle = macro.fn(builder, *handles)
    if strict and len(builder.moves) == before:
        raise MoveError(ERR_DEGENERATE, f"{macro.name} adds no move")
    return handle


def describe(expr: Expr, nested: bool = False) -> str:
    """English description; nested constructions are parenthesised."""
    if isinstance(expr, str):
        return expr
    macro = MACROS[expr[0]]
    text = macro.template.format(*(describe(a, True) for a in expr[1:]))
    return f"({text})" if nested else text


def result_kind(expr: Expr) -> str:
    return POINT if isinstance(expr, str) else MACROS[expr[0]].result


def relabel_expr(expr: Expr, mapping: dict[str, str]) -> Expr:
    if isinstance(expr, str):
        return mapping.get(expr, expr)
    return [expr[0], *(relabel_expr(a, mapping) for a in expr[1:])]


def base_labels(expr: Expr) -> list[str]:
    if isinstance(expr, str):
        return [expr]
    out: list[str] = []
    for a in expr[1:]:
        out.extend(base_labels(a))
    return out


def expected_kind_params(obj: Obj) -> tuple[str, tuple[float, ...]]:
    return obj.kind, obj.params()


def object_from_params(kind: str, params: Sequence[float]) -> Obj:
    if kind == POINT:
        mv = cga.point(params[0], params[1])
    elif kind == LINE:
        mv = cga.line(params[0], params[1], params[2])
    elif kind == CIRCLE:
        mv = cga.circle(params[0], params[1], params[2])
    else:
        raise ValueError(kind)
    return Obj(handle="target", kind=kind, mv=mv)


def transform_params(
    kind: str, params: Sequence[float], g: cga.Similarity
) -> tuple[float, ...]:
    """Euclidean parameters of g(object) — used to map answers between frames."""
    if kind == POINT:
        return g.xy(params[0], params[1])
    if kind == CIRCLE:
        cx, cy = g.xy(params[0], params[1])
        return cx, cy, params[2] * g.scale
    nx, ny, d = params
    # two points on the line: foot of the normal and one step along the direction
    fx, fy = nx * d, ny * d
    tx, ty = ny, -nx
    p = g.xy(fx, fy)
    q = g.xy(fx + tx, fy + ty)
    return cga.line_params(cga.line_through(cga.point(*p), cga.point(*q)))


def inverse_params(
    kind: str, params: Sequence[float], g: cga.Similarity
) -> tuple[float, ...]:
    if kind == POINT:
        return g.inverse_xy(params[0], params[1])
    if kind == CIRCLE:
        cx, cy = g.inverse_xy(params[0], params[1])
        return cx, cy, params[2] / g.scale
    nx, ny, d = params
    fx, fy = nx * d, ny * d
    tx, ty = ny, -nx
    p = g.inverse_xy(fx, fy)
    q = g.inverse_xy(fx + tx, fy + ty)
    return cga.line_params(cga.line_through(cga.point(*p), cga.point(*q)))


def distance_between(
    kind: str, a: Sequence[float], b: Sequence[float], scale: float = 1.0
) -> float:
    """A length-valued distance between two objects' parameters.

    For lines the angle between the normals (its sine) is multiplied by
    `scale` so that it is commensurable with the offset difference; a caller
    comparing against `rel_tol * scale` therefore bounds both the angle (in
    radians, by rel_tol) and the offset (by rel_tol * scale).
    """
    if kind == POINT:
        return math.hypot(a[0] - b[0], a[1] - b[1])
    if kind == CIRCLE:
        return max(math.hypot(a[0] - b[0], a[1] - b[1]), abs(a[2] - b[2]))
    nx, ny, d = b
    if a[0] * nx + a[1] * ny < 0:
        nx, ny, d = -nx, -ny, -d
    return max(abs(a[0] * ny - a[1] * nx) * scale, abs(a[2] - d))
