"""Render a NoReGeo configuration in the benchmark's "dots" style.

Only the falsification arm needs this: it shows the model a figure of a
*scaled* configuration next to the original text question, and both the
perturbed figure and its control have to come out of the same renderer, or the
comparison measures our drawing instead of the model's reading.

Deliberately close to `data/noregeo/images/` — grid, axis lines through the
origin, labelled scatter points, a title that names the task and nothing
else — and deliberately not a copy: what matters is that `swap` and
`swap-control` differ in the configuration alone.

Built on a bare `Figure` rather than pyplot, like `verify.contact_sheet`:
batch runs call this from N worker threads and pyplot's figure registry is
global and unsynchronised.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

# The benchmark's default view. Configurations that do not fit get a fitted box
# instead (`view_box_for`), which is what the benchmark's own images do too.
DEFAULT_VIEW: tuple[float, float, float, float] = (-10.5, 10.5, -10.5, 10.5)

# matplotlib's default property cycle, which is what the benchmark's figures use.
COLORS = (
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
)


def _as_point(value: Any) -> list[float] | None:
    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)
    ):
        return [float(value[0]), float(value[1])]
    return None


def iter_points(points: dict[str, Any]) -> list[tuple[str, list[float]]]:
    """Flatten `named_points()` output to [(label, [x, y])].

    A value that is a list of points (a polygon) contributes its vertices
    unlabelled, which is how the benchmark draws them.
    """
    out: list[tuple[str, list[float]]] = []
    for name, value in points.items():
        single = _as_point(value)
        if single is not None:
            out.append((str(name), single))
            continue
        if isinstance(value, (list, tuple)):
            for item in value:
                vertex = _as_point(item)
                if vertex is not None:
                    out.append(("", vertex))
    return out


def scale_points(points: dict[str, Any], scale: float) -> dict[str, Any]:
    """Scale every coordinate about the origin, preserving the structure."""

    def _walk(value: Any) -> Any:
        single = _as_point(value)
        if single is not None:
            return [single[0] * scale, single[1] * scale]
        if isinstance(value, (list, tuple)):
            return [_walk(item) for item in value]
        return value

    return {name: _walk(value) for name, value in points.items()}


def view_box_for(
    points: dict[str, Any], pad: float = 1.0
) -> tuple[float, float, float, float]:
    """The view box to draw in: the benchmark's default when everything fits.

    A fitted box otherwise — some configurations genuinely leave [-10.5, 10.5]
    (the bisector tasks reach y = -19).
    """
    flat = [xy for _, xy in iter_points(points)]
    if not flat:
        return DEFAULT_VIEW
    xs = [p[0] for p in flat]
    ys = [p[1] for p in flat]
    x0, x1, y0, y1 = DEFAULT_VIEW
    if min(xs) >= x0 and max(xs) <= x1 and min(ys) >= y0 and max(ys) <= y1:
        return DEFAULT_VIEW
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    margin = max(pad, 0.1 * span)
    return (
        round(min(xs) - margin, 2),
        round(max(xs) + margin, 2),
        round(min(ys) - margin, 2),
        round(max(ys) + margin, 2),
    )


def render_dots(
    points: dict[str, Any],
    title: str,
    out_path: str | Path,
    view: Sequence[float] | None = None,
    figsize: tuple[float, float] = (4.4, 4.4),
    dpi: int = 100,
) -> Path:
    """Draw labelled points on a grid and save a PNG.

    `view` is passed in rather than derived so that a perturbed configuration
    and its control share one set of axes: with a fitted view the scaled figure
    would look pixel-identical to the control except for the tick labels, and
    the falsification test would be a tick-reading test.
    """
    from matplotlib.figure import Figure

    x0, x1, y0, y1 = view if view is not None else view_box_for(points)
    fig = Figure(figsize=figsize, dpi=dpi)
    ax = fig.subplots()
    ax.set_aspect("equal")
    ax.grid(True, ls=":", lw=0.6, alpha=0.6)
    ax.axhline(0.0, color="tab:blue", lw=0.9, zorder=0)
    ax.axvline(0.0, color="tab:blue", lw=0.9, zorder=0)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)

    span = max(x1 - x0, y1 - y0)
    offset = 0.02 * span
    for i, (label, (x, y)) in enumerate(iter_points(points)):
        ax.scatter([x], [y], s=28, color=COLORS[i % len(COLORS)], zorder=5)
        if label:
            ax.annotate(
                label,
                (x + offset, y + offset),
                fontsize=10,
                color="black",
                zorder=6,
            )
    ax.set_title(title, fontsize=11)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, facecolor="white")
    return out_path
