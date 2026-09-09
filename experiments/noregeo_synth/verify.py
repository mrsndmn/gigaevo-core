"""Checks that decide whether a generated sample is usable.

Three independent signals:

* the code ran (`execute.py`),
* every step actually changed the canvas (`pixel_deltas`) — this is what
  enforces "incremental", as opposed to steps that merely claim to draw,
* the final answer matches the benchmark's golden answer (`answer_matches`).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np


def answer_matches(row: dict[str, Any], final: Any, cfg) -> tuple[bool, str]:
    """Compare a model answer against the benchmark's golden answer."""
    gold = row.get("answer")
    schema_type = row.get("schema_type")

    if schema_type == "classification":
        got = str(final).strip().lower()
        want = str(gold).strip().lower()
        return got == want, f"{final!r} vs golden {gold!r}"

    if schema_type == "point":
        want_xy = gold.get("point") if isinstance(gold, dict) else gold
        if isinstance(final, dict):
            got_xy = final.get("point", [final.get("x"), final.get("y")])
        else:
            got_xy = final
        try:
            dx = float(got_xy[0]) - float(want_xy[0])
            dy = float(got_xy[1]) - float(want_xy[1])
        except (TypeError, ValueError, IndexError):
            return False, f"unparseable point {final!r}"
        dist = math.hypot(dx, dy)
        return dist <= cfg.point_atol, (
            f"({got_xy[0]:.4f}, {got_xy[1]:.4f}) vs golden "
            f"({float(want_xy[0]):.4f}, {float(want_xy[1]):.4f}), |d|={dist:.4g}"
        )

    try:
        got = float(final)
        want = float(gold)
    except (TypeError, ValueError):
        return False, f"unparseable number {final!r}"
    ok = math.isclose(got, want, rel_tol=cfg.num_rtol, abs_tol=cfg.num_atol)
    return ok, f"{got:g} vs golden {want:g}"


def pixel_deltas(images: list[str | Path]) -> list[float]:
    """Fraction of pixels each step changed w.r.t. the previous step's PNG.

    Element 0 has no predecessor, so it reports the ink the first step put down
    on the (white) canvas instead. A delta at or near zero means the step drew
    nothing new — the failure mode this check exists to catch.
    """
    import matplotlib.image as mpimg

    deltas: list[float] = []
    prev: np.ndarray | None = None
    for path in images:
        arr = mpimg.imread(str(path))
        arr = arr[..., :3] if arr.ndim == 3 else arr
        if prev is None:
            # vs. a white canvas: how much ink the first step put down.
            changed = float(np.mean(np.abs(arr - 1.0) > 0.02))
        elif prev.shape != arr.shape:
            changed = float("nan")
        else:
            changed = float(np.mean(np.abs(arr - prev) > 0.02))
        deltas.append(changed)
        prev = arr
    return deltas


def contact_sheet(
    images: list[str | Path],
    titles: list[str],
    out_path: Path,
    suptitle: str = "",
) -> Path:
    """Montage of the per-step PNGs, for eyeballing a sample at a glance."""
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    n = len(images)
    cols = min(3, n) or 1
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 5.0 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(mpimg.imread(str(img)))
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, facecolor="white")
    plt.close(fig)
    return out_path
