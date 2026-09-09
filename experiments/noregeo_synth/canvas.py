"""Build the shared matplotlib canvas the generated steps draw onto.

The preamble is generated *by us*, not by the model: it fixes the figure, the
axes, the view box and the named points, so that every step snippet is a pure
increment on a known environment. The exact source is also pasted into the
prompt, so the model sees the environment it is coding against.
"""

from __future__ import annotations

from typing import Any

from .configs import GenConfig
from .dataset import coordinate_pairs, named_points

PREAMBLE_TEMPLATE = '''\
import math

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Named points of the task (numpy arrays; a polygon is an (N, 2) array).
P = {points_src}

# Scratch dict: put computed values here so later steps can reuse them.
state = {{}}

fig, ax = plt.subplots(figsize=({figw}, {figh}), dpi={dpi})
ax.set_aspect("equal")
ax.grid(True, ls=":", lw=0.6, alpha=0.6)
ax.axhline(0.0, color="0.55", lw=0.9, zorder=0)
ax.axvline(0.0, color="0.55", lw=0.9, zorder=0)
ax.set_xlim({x0}, {x1})
ax.set_ylim({y0}, {y1})
ax.set_xlabel("x")
ax.set_ylabel("y")


def expand_view(*points, margin=1.0):
    """Grow the axes limits so every given point stays visible."""
    xs, ys = [], []
    for p in points:
        arr = np.asarray(p, dtype=float).reshape(-1, 2)
        xs.extend(arr[:, 0].tolist())
        ys.extend(arr[:, 1].tolist())
    if not xs:
        return
    lo_x, hi_x = ax.get_xlim()
    lo_y, hi_y = ax.get_ylim()
    ax.set_xlim(min(lo_x, min(xs) - margin), max(hi_x, max(xs) + margin))
    ax.set_ylim(min(lo_y, min(ys) - margin), max(hi_y, max(ys) + margin))


def mark(name, xy, color="tab:blue", size=55, dxy=(0.3, 0.3)):
    """Scatter one point, label it, and keep it inside the view."""
    xy = np.asarray(xy, dtype=float)
    ax.scatter([xy[0]], [xy[1]], s=size, color=color, zorder=5)
    ax.annotate(
        name,
        (xy[0] + dxy[0], xy[1] + dxy[1]),
        color=color,
        fontsize=11,
        fontweight="bold",
        zorder=6,
    )
    expand_view(xy, margin=1.0)
    return xy
'''


def _points_src(row: dict[str, Any]) -> str:
    pts = named_points(row)
    if not pts:
        return "{}"
    items = ", ".join(f'"{k}": np.array({v}, dtype=float)' for k, v in pts.items())
    return "{" + items + "}"


def view_box(row: dict[str, Any], pad_frac: float = 0.18) -> tuple[float, ...]:
    pairs = coordinate_pairs(row)
    if not pairs:
        return (-10.0, 10.0, -10.0, 10.0)
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    pad = max(1.0, pad_frac * span)
    return (
        round(min(xs) - pad, 2),
        round(max(xs) + pad, 2),
        round(min(ys) - pad, 2),
        round(max(ys) + pad, 2),
    )


def build_preamble(row: dict[str, Any], cfg: GenConfig) -> str:
    x0, x1, y0, y1 = view_box(row)
    return PREAMBLE_TEMPLATE.format(
        points_src=_points_src(row),
        figw=cfg.figwidth,
        figh=cfg.figheight,
        dpi=cfg.dpi,
        x0=x0,
        x1=x1,
        y0=y0,
        y1=y1,
    )
