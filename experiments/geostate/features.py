"""Exact invariant features of a geometric state — the input of the V1 encoder.

For the object being appended and every object that already exists, a fixed
vector of quantities that an orientation-preserving similarity cannot change:
kinds, distances divided by the state's scale, radii over scale, the sine and
cosine of the angle between two lines, signed (left/right) point-to-line
distances, incidence and tangency indicators. No coordinate enters, so the
state token is invariant by construction (interview Round 6: the
invariant-feature encoder replaces a learned equivariant network in the
probe). Everything here is numpy; `encoder.py` turns it into torch tensors.
"""

from __future__ import annotations

import math

import numpy as np

from .configs import TOL, ToleranceSpec
from .state import CIRCLE, LINE, POINT, GeometricState, Obj

KIND_INDEX = {POINT: 0, LINE: 1, CIRCLE: 2}
SELF_DIM = 5  # kind one-hot (3), radius/scale, is_given
PAIR_DIM = 16  # see `pair_features`


def _kind_onehot(kind: str) -> list[float]:
    v = [0.0, 0.0, 0.0]
    v[KIND_INDEX[kind]] = 1.0
    return v


def self_features(obj: Obj, scale: float) -> np.ndarray:
    radius = obj.params()[2] / scale if obj.kind == CIRCLE else 0.0
    return np.array(
        [*_kind_onehot(obj.kind), radius, 1.0 if obj.move_index < 0 else 0.0],
        dtype=np.float32,
    )


def _signed_point_line(px: float, py: float, nx: float, ny: float, d: float) -> float:
    return nx * px + ny * py - d


def pair_features(a: Obj, b: Obj, scale: float, tol: ToleranceSpec = TOL) -> np.ndarray:
    """Invariant relation of `a` (the new object) to `b` (an existing one).

    Layout (16): kind(a) 3 | kind(b) 3 | d1 d2 d3 d4 (kind-pair specific,
    scale-free) | cos sin (line-line) | incident | tangent | same | reserved.
    """
    f = np.zeros(PAIR_DIM, dtype=np.float32)
    f[0:3] = _kind_onehot(a.kind)
    f[3:6] = _kind_onehot(b.kind)
    pa, pb = a.params(), b.params()
    eps = tol.incidence * scale
    ka, kb = a.kind, b.kind
    if ka == POINT and kb == POINT:
        d = math.hypot(pa[0] - pb[0], pa[1] - pb[1]) / scale
        f[6] = d
        f[14] = 1.0 if d * scale <= eps else 0.0
    elif {ka, kb} == {POINT, LINE}:
        p, line_obj = (pa, pb) if ka == POINT else (pb, pa)
        s = _signed_point_line(p[0], p[1], line_obj[0], line_obj[1], line_obj[2])
        f[6] = s / scale
        f[7] = abs(s) / scale
        f[12] = 1.0 if abs(s) <= eps else 0.0
    elif {ka, kb} == {POINT, CIRCLE}:
        p, c = (pa, pb) if ka == POINT else (pb, pa)
        dist = math.hypot(p[0] - c[0], p[1] - c[1])
        f[6] = dist / scale
        f[7] = (dist - c[2]) / scale
        f[12] = 1.0 if abs(dist - c[2]) <= eps else 0.0
    elif ka == LINE and kb == LINE:
        # Only the angle is a clean invariant of two lines; the offset between
        # parallel lines would need a parallelism threshold that float error
        # can flip, and it is recoverable from the points on the lines anyway.
        cos = pa[0] * pb[0] + pa[1] * pb[1]
        sin = pa[0] * pb[1] - pa[1] * pb[0]
        f[10] = cos
        f[11] = sin
        same_dir = pb if cos >= 0 else (-pb[0], -pb[1], -pb[2])
        f[14] = (
            1.0
            if abs(sin) <= tol.coincidence and abs(pa[2] - same_dir[2]) <= eps
            else 0.0
        )
    elif {ka, kb} == {LINE, CIRCLE}:
        line_obj, c = (pa, pb) if ka == LINE else (pb, pa)
        s = _signed_point_line(c[0], c[1], line_obj[0], line_obj[1], line_obj[2])
        f[6] = s / scale
        f[7] = (abs(s) - c[2]) / scale
        f[8] = c[2] / scale
        f[13] = 1.0 if abs(abs(s) - c[2]) <= eps else 0.0
    else:  # circle-circle
        dist = math.hypot(pa[0] - pb[0], pa[1] - pb[1])
        f[6] = dist / scale
        f[7] = (pa[2] - pb[2]) / scale
        f[8] = (dist - pa[2] - pb[2]) / scale
        f[9] = (dist - abs(pa[2] - pb[2])) / scale
        f[13] = (
            1.0
            if abs(dist - pa[2] - pb[2]) <= eps or abs(dist - abs(pa[2] - pb[2])) <= eps
            else 0.0
        )
        f[14] = 1.0 if dist <= eps and abs(pa[2] - pb[2]) <= eps else 0.0
    return f


def object_features(
    state: GeometricState, handle: str, context: list[str] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Self vector of `handle` and the pair matrix against `context` handles.

    `context` defaults to every object that precedes `handle` in the state,
    plus the object itself, so that a token never depends on objects appended
    later (append-only, KV-cache friendly).
    """
    obj = state.get(handle)
    if context is None:
        seen = []
        for o in state.objects:
            seen.append(o.handle)
            if o.handle == handle:
                break
        context = seen
    pairs = np.stack([pair_features(obj, state.get(h), state.scale) for h in context])
    return self_features(obj, state.scale), pairs


def state_token_inputs(
    state: GeometricState, handles: list[str]
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Features for the state tokens of `handles`, in order."""
    return [object_features(state, h) for h in handles]
