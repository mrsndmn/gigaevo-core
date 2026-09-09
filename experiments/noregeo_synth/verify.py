"""Checks that decide whether a generated sample is usable.

Signals, all independent:

* the code ran (`execute.py`);
* every step changed the canvas *relative to the frame before it* — including
  step 1, which is compared against the rendered preamble, not against a blank
  image (grid and axes alone move ~3% of the pixels, so a blank baseline would
  wave through a step that draws nothing);
* no step erased what earlier steps drew (`ink_loss`), checked only while the
  view box is unchanged, since `expand_view` legitimately moves every pixel;
* the final answer matches the golden answer, both as prose (`final_answer`)
  and as something the code itself computed (`state["answer"]`);
* the golden answer was not simply pasted into the code as a literal
  (`literal_leak`) — with `reveal_answer=True` the answer is in the prompt, so
  this is what keeps "the code computed it" from meaning "the code retyped it".
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import given_numbers

INK_THRESHOLD = 0.02  # distance from white that counts as "something is drawn"


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
            got_x, got_y = float(got_xy[0]), float(got_xy[1])
            want_x, want_y = float(want_xy[0]), float(want_xy[1])
        except (TypeError, ValueError, IndexError):
            return False, f"unparseable point {final!r}"
        dist = math.hypot(got_x - want_x, got_y - want_y)
        return dist <= cfg.point_atol, (
            f"({got_x:.4f}, {got_y:.4f}) vs golden ({want_x:.4f}, {want_y:.4f}), "
            f"|d|={dist:.4g}"
        )

    try:
        got = float(final)
        want = float(gold)
    except (TypeError, ValueError):
        return False, f"unparseable number {final!r}"
    ok = math.isclose(got, want, rel_tol=cfg.num_rtol, abs_tol=cfg.num_atol)
    return ok, f"{got:g} vs golden {want:g}"


# ---------------------------------------------------------------------------
# Frame-to-frame comparison
# ---------------------------------------------------------------------------


def _load(path: str | Path) -> np.ndarray:
    import matplotlib.image as mpimg

    arr = mpimg.imread(str(path))
    return arr[..., :3] if arr.ndim == 3 else arr


def _ink(arr: np.ndarray) -> np.ndarray:
    diff = np.abs(arr - 1.0)
    return (diff.max(axis=-1) if diff.ndim == 3 else diff) > INK_THRESHOLD


def _common_box(a: list[int] | None, b: list[int] | None) -> list[int] | None:
    """Intersection of two axes rectangles.

    With `set_aspect("equal")` the axes box itself shrinks when the view box
    grows, so consecutive frames do not always share one rectangle; comparing
    their overlap keeps the two views the same shape.
    """
    if not a or not b:
        return None
    left, right = max(a[0], b[0]), min(a[1], b[1])
    top, bottom = max(a[2], b[2]), min(a[3], b[3])
    if right <= left or bottom <= top:
        return None
    return [left, right, top, bottom]


def _region_mask(
    shape: tuple[int, ...], include: list[int], exclude: list[list[int] | None]
) -> np.ndarray:
    """Pixels that count: inside the axes rectangle, outside the legends."""
    mask = np.zeros(shape[:2], dtype=bool)
    left, right, top, bottom = include
    mask[max(top, 0) : max(bottom, 0), max(left, 0) : max(right, 0)] = True
    for box in exclude:
        if box:
            left, right, top, bottom = box
            mask[max(top, 0) : max(bottom, 0), max(left, 0) : max(right, 0)] = False
    return mask


def compare_frames(
    frames: list[dict], baseline: dict | None = None
) -> list[dict[str, float | None]]:
    """Per-step `{delta, ink_loss}` against the frame that preceded it.

    Each frame is `{"image", "view_box", "axes_box", "legend_box"}` as reported
    by `_runner.py`. Comparison happens inside the axes rectangle and outside
    the legend: the step title above the axes changes on every step, and a
    `loc="best"` legend jumps corners whenever a labelled artist is added —
    both would otherwise read as "this step drew something" (and the legend
    additionally as "this step erased something").

    `delta` is the fraction of pixels that changed; `ink_loss` the fraction of
    previously-drawn pixels that went back to white. Either is None when the
    two frames cannot be compared honestly — different pixel dimensions, no
    overlapping axes rectangle, or (for `ink_loss`) a view box that moved.
    """
    out: list[dict[str, float | None]] = []
    prev = baseline
    prev_arr = _load(baseline["image"]) if baseline else None

    for frame in frames:
        arr = _load(frame["image"])
        entry: dict[str, float | None]
        if prev_arr is None:
            entry = {"delta": float(np.mean(_ink(arr))), "ink_loss": None}
        elif prev_arr.shape != arr.shape:
            entry = {"delta": None, "ink_loss": None}
        else:
            box = _common_box(frame.get("axes_box"), prev.get("axes_box"))
            mask = (
                _region_mask(
                    arr.shape, box, [frame.get("legend_box"), prev.get("legend_box")]
                )
                if box
                else None
            )
            if mask is None or not mask.any():
                entry = {"delta": None, "ink_loss": None}
            else:
                changed = np.abs(arr - prev_arr)
                changed = changed.max(axis=-1) if changed.ndim == 3 else changed
                delta = float(np.mean(changed[mask] > INK_THRESHOLD))
                loss: float | None = None
                if frame.get("view_box") and frame["view_box"] == prev.get("view_box"):
                    prev_ink = _ink(prev_arr) & mask
                    total = int(prev_ink.sum())
                    lost = int(np.logical_and(prev_ink, ~_ink(arr)).sum())
                    loss = lost / total if total else 0.0
                entry = {"delta": delta, "ink_loss": loss}
        out.append(entry)
        prev, prev_arr = frame, arr
    return out


# ---------------------------------------------------------------------------
# Pasted-answer detection
# ---------------------------------------------------------------------------


def _numeric_literals(code: str) -> list[float]:
    """Numeric literals in the code, with unary minus folded in.

    Without the folding `np.array([3.46, -7.0])` looks like it contains `7.0`,
    which is a *given* coordinate of the task — so a pasted answer would read
    as legitimate reuse of the inputs.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    negated: set[int] = set()
    out: list[float] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, (int, float))
            and not isinstance(node.operand.value, bool)
        ):
            negated.add(id(node.operand))
            out.append(-float(node.operand.value))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
            and id(node) not in negated
        ):
            out.append(float(node.value))
    return out


def literal_leak(
    row: dict[str, Any], steps: list[dict], atol: float = 1e-6
) -> list[str]:
    """Steps that hard-code the golden answer instead of deriving it.

    Numbers that already appear in the problem statement are not leaks — the
    code is told to use the given coordinates. For point answers both
    coordinates have to show up in the same step before it counts, so a step
    that happens to contain one matching number is not accused.
    """
    schema_type = row.get("schema_type")
    gold = row.get("answer")
    if schema_type == "point":
        xy = gold.get("point") if isinstance(gold, dict) else gold
        try:
            targets = [float(xy[0]), float(xy[1])]
        except (TypeError, ValueError, IndexError):
            return []
        need_all = True
    elif schema_type == "number":
        try:
            targets = [float(gold)]
        except (TypeError, ValueError):
            return []
        need_all = False
    else:
        return []

    given = given_numbers(row)

    def _is_given(value: float) -> bool:
        return any(abs(value - g) <= atol for g in given)

    hits: list[str] = []
    for i, step in enumerate(steps, 1):
        lits = _numeric_literals(step.get("code", ""))
        matched = [
            t
            for t in targets
            if not _is_given(t)
            and any(abs(lit - t) <= atol * max(1.0, abs(t)) for lit in lits)
        ]
        wanted = [t for t in targets if not _is_given(t)]
        if not wanted:
            continue
        if (need_all and len(matched) == len(wanted)) or (not need_all and matched):
            hits.append(
                f"step {i} hard-codes the answer as a literal "
                f"({', '.join(f'{m:g}' for m in matched)})"
            )
    return hits


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


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
