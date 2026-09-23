"""Geometric state, moves and goal predicates — the environment's core.

A `GeometricState` is the set of objects a construction has produced: for
each, a handle, a kind (point / line / circle), its CGA multivector and an
optional label. Given points are labelled and their handle *is* the label;
constructed objects get sequential handles `h1, h2, ...` in the order they are
appended. A `Move` is one primitive: `line(P, Q)`, `circle(O, P)` or
`meet(U, V)`. The transition is exact (kernel operations on multivectors) and
appends; nothing is ever mutated or re-read from pixels.

Intersection pairs are ordered by orientation (ADR 0005):
  circle & circle: first the point on the left of the directed line from the
                   centre of the first circle to the centre of the second;
  circle & line:   first the point that comes first along the line's direction.
Both rules are preserved by every orientation-preserving similarity.

Every tolerance is relative to `state.scale`, the diameter of the given points,
so that predicates and degeneracy checks are scale-invariant.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import math

import numpy as np

from . import cga
from .configs import TOL, ToleranceSpec

POINT = "point"
LINE = "line"
CIRCLE = "circle"
KINDS = (POINT, LINE, CIRCLE)

OP_LINE = "line"
OP_CIRCLE = "circle"
OP_MEET = "meet"
OPS = (OP_LINE, OP_CIRCLE, OP_MEET)

# Error kinds, in the vocabulary GeoBuildBench uses for its failure analysis.
ERR_UNDEFINED = "undefined_reference"
ERR_TYPE = "type"
ERR_DEGENERATE = "degenerate"


class MoveError(ValueError):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True)
class Move:
    op: str
    args: tuple[str, ...]

    def text(self) -> str:
        return f"{self.op}({', '.join(self.args)})"

    def to_dict(self) -> dict:
        return {"op": self.op, "args": list(self.args)}

    @classmethod
    def from_dict(cls, d: Mapping) -> Move:
        return cls(op=str(d["op"]), args=tuple(str(a) for a in d["args"]))


@dataclass
class Obj:
    handle: str
    kind: str
    mv: np.ndarray
    label: str | None = None
    # index of the move that produced it; -1 for given objects
    move_index: int = -1

    def xy(self) -> tuple[float, float]:
        if self.kind != POINT:
            raise TypeError(f"{self.handle} is a {self.kind}, not a point")
        return cga.to_xy(self.mv)

    def params(self) -> tuple[float, ...]:
        if self.kind == POINT:
            return cga.to_xy(self.mv)
        if self.kind == LINE:
            return cga.line_params(self.mv)
        return cga.circle_params(self.mv)

    def describe(self, digits: int = 3) -> str:
        p = self.params()
        if self.kind == POINT:
            return f"point ({p[0]:.{digits}f}, {p[1]:.{digits}f})"
        if self.kind == LINE:
            return f"line n=({p[0]:.{digits}f}, {p[1]:.{digits}f}) d={p[2]:.{digits}f}"
        return (
            f"circle centre ({p[0]:.{digits}f}, {p[1]:.{digits}f}) r={p[2]:.{digits}f}"
        )


def _cross(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * by - ay * bx


def same_geometry(a: Obj, b: Obj, scale: float, rel_tol: float) -> bool:
    """Do two objects coincide, within rel_tol * scale?"""
    if a.kind != b.kind:
        return False
    eps = rel_tol * scale
    pa, pb = a.params(), b.params()
    if a.kind == POINT:
        return math.hypot(pa[0] - pb[0], pa[1] - pb[1]) <= eps
    if a.kind == CIRCLE:
        return (
            math.hypot(pa[0] - pb[0], pa[1] - pb[1]) <= eps
            and abs(pa[2] - pb[2]) <= eps
        )
    dot = pa[0] * pb[0] + pa[1] * pb[1]
    if dot >= 0:
        return abs(1.0 - dot) <= rel_tol and abs(pa[2] - pb[2]) <= eps
    return abs(1.0 + dot) <= rel_tol and abs(pa[2] + pb[2]) <= eps


def incident(p: Obj, o: Obj, scale: float, rel_tol: float) -> bool:
    """Is point p on line/circle o?"""
    if p.kind != POINT or o.kind == POINT:
        raise TypeError("incident(point, line|circle)")
    x, y = p.xy()
    eps = rel_tol * scale
    if o.kind == LINE:
        nx, ny, d = o.params()
        return abs(nx * x + ny * y - d) <= eps
    cx, cy, r = o.params()
    return abs(math.hypot(x - cx, y - cy) - r) <= eps


def tangent(a: Obj, b: Obj, scale: float, rel_tol: float) -> bool:
    """Are two circles, or a line and a circle, tangent?"""
    if POINT in (a.kind, b.kind) or (a.kind == LINE and b.kind == LINE):
        raise TypeError("tangent(circle|line, circle|line) with at least one circle")
    eps = rel_tol * scale
    if a.kind == CIRCLE and b.kind == CIRCLE:
        ax, ay, ar = a.params()
        bx, by, br = b.params()
        d = math.hypot(ax - bx, ay - by)
        return abs(d - (ar + br)) <= eps or abs(d - abs(ar - br)) <= eps
    line_obj, circ = (a, b) if a.kind == LINE else (b, a)
    nx, ny, d = line_obj.params()
    cx, cy, r = circ.params()
    return abs(abs(nx * cx + ny * cy - d) - r) <= eps


@dataclass
class GeometricState:
    tol: ToleranceSpec = TOL
    objects: list[Obj] = field(default_factory=list)
    _index: dict[str, Obj] = field(default_factory=dict, repr=False)
    _n_constructed: int = 0
    _n_moves: int = 0
    scale: float = 1.0

    @classmethod
    def from_given(
        cls, given: Mapping[str, tuple[float, float]], tol: ToleranceSpec = TOL
    ) -> GeometricState:
        if len(given) < 2:
            raise ValueError("a state needs at least two given points")
        state = cls(tol=tol)
        for label, (x, y) in given.items():
            state._add(Obj(handle=label, kind=POINT, mv=cga.point(x, y), label=label))
        state.scale = state._diameter()
        return state

    # -- bookkeeping ---------------------------------------------------------

    @property
    def given(self) -> list[Obj]:
        return [o for o in self.objects if o.move_index < 0]

    @property
    def n_moves(self) -> int:
        return self._n_moves

    def _diameter(self) -> float:
        pts = [o.xy() for o in self.given]
        best = 0.0
        for i, (ax, ay) in enumerate(pts):
            for bx, by in pts[i + 1 :]:
                best = max(best, math.hypot(ax - bx, ay - by))
        if best <= 0.0:
            raise ValueError("given points coincide")
        return best

    def _add(self, obj: Obj) -> Obj:
        if obj.handle in self._index:
            raise ValueError(f"duplicate handle {obj.handle}")
        self.objects.append(obj)
        self._index[obj.handle] = obj
        return obj

    def _next_handle(self) -> str:
        self._n_constructed += 1
        return f"h{self._n_constructed}"

    def __contains__(self, handle: str) -> bool:
        return handle in self._index

    def get(self, handle: str) -> Obj:
        try:
            return self._index[handle]
        except KeyError:
            raise MoveError(ERR_UNDEFINED, f"unknown object {handle!r}") from None

    def handles(self) -> list[str]:
        return [o.handle for o in self.objects]

    def find(self, kind: str, mv: np.ndarray) -> Obj | None:
        """An existing object with this geometry, if any."""
        probe = Obj(handle="?", kind=kind, mv=mv)
        for o in self.objects:
            if o.kind == kind and same_geometry(
                o, probe, self.scale, self.tol.coincidence
            ):
                return o
        return None

    def copy(self) -> GeometricState:
        clone = GeometricState(tol=self.tol, scale=self.scale)
        for o in self.objects:
            clone._add(Obj(o.handle, o.kind, o.mv.copy(), o.label, o.move_index))
        clone._n_constructed = self._n_constructed
        clone._n_moves = self._n_moves
        return clone

    # -- the transition -----------------------------------------------------

    def _point_arg(self, handle: str) -> Obj:
        o = self.get(handle)
        if o.kind != POINT:
            raise MoveError(ERR_TYPE, f"{handle} is a {o.kind}, expected a point")
        return o

    def _curve_arg(self, handle: str) -> Obj:
        o = self.get(handle)
        if o.kind == POINT:
            raise MoveError(ERR_TYPE, f"{handle} is a point, expected a line or circle")
        return o

    def compute(self, move: Move) -> list[tuple[str, np.ndarray]]:
        """The objects a move would produce, without appending them."""
        if move.op not in OPS:
            raise MoveError(ERR_TYPE, f"unknown primitive {move.op!r}")
        if len(move.args) != 2:
            raise MoveError(ERR_TYPE, f"{move.op} takes two arguments")
        sep = self.tol.min_point_separation * self.scale
        if move.op == OP_LINE:
            p, q = self._point_arg(move.args[0]), self._point_arg(move.args[1])
            if math.dist(p.xy(), q.xy()) < sep:
                raise MoveError(
                    ERR_DEGENERATE,
                    f"{move.text()}: points too close (below the minimum separation)",
                )
            return [(LINE, cga.line_through(p.mv, q.mv))]
        if move.op == OP_CIRCLE:
            c, p = self._point_arg(move.args[0]), self._point_arg(move.args[1])
            if math.dist(c.xy(), p.xy()) < sep:
                raise MoveError(
                    ERR_DEGENERATE, f"{move.text()}: zero radius (points too close)"
                )
            return [(CIRCLE, cga.circle_through(c.mv, p.mv))]
        a, b = self._curve_arg(move.args[0]), self._curve_arg(move.args[1])
        return [(POINT, mv) for mv in self._meet(a, b, move)]

    def _meet(self, a: Obj, b: Obj, move: Move) -> list[np.ndarray]:
        if same_geometry(a, b, self.scale, self.tol.coincidence):
            raise MoveError(ERR_DEGENERATE, f"{move.text()}: identical objects")
        T, s = cga.meet_pair(a.mv, b.mv)
        if a.kind == LINE and b.kind == LINE:
            ta, tb = cga.line_direction(a.mv), cga.line_direction(b.mv)
            angle = math.degrees(math.asin(min(1.0, abs(_cross(*ta, *tb)))))
            xy = cga.flat_point_xy(T)
            if xy is None or angle < self.tol.min_line_angle_deg:
                raise MoveError(ERR_DEGENERATE, f"{move.text()}: lines are parallel")
            return [cga.point(*xy)]
        if s <= 0.0:
            # T.T = (|p-q|^2 / 2)^2 for a real pair; |T.T| estimates the gap
            # between the two (possibly imaginary) points.
            gap = math.sqrt(2.0 * math.sqrt(abs(s)))
            if gap < self.tol.min_meet_separation * self.scale:
                raise MoveError(
                    ERR_DEGENERATE, f"{move.text()}: objects are (nearly) tangent"
                )
            raise MoveError(ERR_DEGENERATE, f"{move.text()}: objects do not intersect")
        P, Q = cga.pair_points(T, s)
        p, q = cga.to_xy(P), cga.to_xy(Q)
        if math.dist(p, q) < self.tol.min_meet_separation * self.scale:
            raise MoveError(
                ERR_DEGENERATE, f"{move.text()}: objects are (nearly) tangent"
            )
        first, second = self._order_pair(a, b, p, q)
        return [cga.point(*first), cga.point(*second)]

    @staticmethod
    def _order_pair(
        a: Obj, b: Obj, p: tuple[float, float], q: tuple[float, float]
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        if a.kind == CIRCLE and b.kind == CIRCLE:
            ax, ay, _ = a.params()
            bx, by, _ = b.params()
            key_p = _cross(bx - ax, by - ay, p[0] - ax, p[1] - ay)
            key_q = _cross(bx - ax, by - ay, q[0] - ax, q[1] - ay)
            return (p, q) if key_p > key_q else (q, p)
        line_obj = a if a.kind == LINE else b
        tx, ty = cga.line_direction(line_obj.mv)
        return (p, q) if tx * p[0] + ty * p[1] <= tx * q[0] + ty * q[1] else (q, p)

    def apply(self, move: Move) -> list[Obj]:
        """Execute a move: validate, compute, append. Raises MoveError."""
        produced = self.compute(move)
        index = self._n_moves
        self._n_moves += 1
        out = []
        for kind, mv in produced:
            out.append(self._add(Obj(self._next_handle(), kind, mv, None, index)))
        return out

    def replay(self, moves: Iterable[Move]) -> list[list[Obj]]:
        return [self.apply(m) for m in moves]

    # -- symmetries ---------------------------------------------------------

    def transform(self, g: cga.Similarity) -> GeometricState:
        """g applied to every object; handles, labels and move indices kept."""
        V = g.versor()
        clone = GeometricState(tol=self.tol)
        for o in self.objects:
            mv = cga.apply(V, o.mv, odd=g.reflect)
            if o.kind == POINT:
                mv = cga.normalize_point(mv)
            elif o.kind == CIRCLE:
                mv = cga.normalize_circle(mv)
            else:
                tx, ty = g.linear(*cga.line_direction(o.mv))
                mv = cga.oriented_line(mv, tx, ty)
            clone._add(Obj(o.handle, o.kind, mv, o.label, o.move_index))
        clone._n_constructed = self._n_constructed
        clone._n_moves = self._n_moves
        clone.scale = clone._diameter()
        return clone

    def relabel(self, mapping: Mapping[str, str]) -> GeometricState:
        """Rename given points (handle = label); constructed handles unchanged."""
        clone = GeometricState(tol=self.tol, scale=self.scale)
        for o in self.objects:
            if o.move_index < 0:
                new = mapping.get(o.handle, o.handle)
                clone._add(Obj(new, o.kind, o.mv.copy(), new, o.move_index))
            else:
                clone._add(Obj(o.handle, o.kind, o.mv.copy(), o.label, o.move_index))
        clone._n_constructed = self._n_constructed
        clone._n_moves = self._n_moves
        return clone

    # -- readouts -----------------------------------------------------------

    def coincides(self, a: str, b: str) -> bool:
        return same_geometry(self.get(a), self.get(b), self.scale, self.tol.coincidence)

    def contains_geometry(self, kind: str, mv: np.ndarray) -> Obj | None:
        return self.find(kind, mv)

    def table(self, digits: int = 3) -> str:
        """Textual state table — what the matched baseline sees instead of tokens."""
        rows = []
        for o in self.objects:
            name = o.handle if o.label is None else o.label
            rows.append(f"{name}: {o.describe(digits)}")
        return "\n".join(rows)


def relabel_move(move: Move, mapping: Mapping[str, str]) -> Move:
    return Move(move.op, tuple(mapping.get(a, a) for a in move.args))
