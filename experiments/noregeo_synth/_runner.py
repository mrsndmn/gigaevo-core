"""Subprocess entrypoint: execute preamble + step snippets, one PNG per step.

Run as ``python -I _runner.py job.json`` — `-I` drops PYTHONPATH and the user
site dir, so this file must stay self-contained (stdlib + matplotlib/numpy from
the active env). It prints one JSON object on stdout.

The whole point of a single process for the whole sequence is incrementality:
the namespace (and therefore the figure) persists across steps, so step k's PNG
contains everything steps 1..k-1 drew.

Besides the images it reports what the caller needs to judge whether the sample
is honest:

* ``base_frame`` — the canvas with the preamble alone, so step 1 can be checked
  for actually drawing something;
* per-step ``view_box`` (data limits), ``axes_box`` and ``legend_box`` (pixel
  rectangles), so the comparison can tell "the picture lost ink" from
  "expand_view moved every pixel" and from "the legend jumped corners";
* ``state_answer`` — what the code itself computed;
* ``answer_point_distance`` — how far the nearest *marked* point is from the
  benchmark's golden point. A trace whose prose is right but whose drawing
  lands somewhere else is worse than no trace at all.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import traceback


def _jsonable(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def _view_box(ax):
    return [float(v) for v in (*ax.get_xlim(), *ax.get_ylim())]


def _axes_box_px(fig, ax):
    """The axes rectangle in image pixels: [left, right, top, bottom].

    Frames are compared inside this rectangle only. The per-step title lives
    outside it, and it changes on every step ("Step 1.", "Step 2.", ...) — a
    full-frame diff would therefore report a change for a step that drew
    nothing at all, which is exactly the failure the diff exists to catch.
    Call after `savefig`, when a renderer exists.
    """
    bbox = ax.get_window_extent()
    height = fig.get_window_extent().height
    return [
        int(round(bbox.x0)),
        int(round(bbox.x1)),
        int(round(height - bbox.y1)),
        int(round(height - bbox.y0)),
    ]


def _legend_box_px(fig, ax):
    """The legend rectangle in image pixels, or None.

    Excluded from frame comparison: with `loc="best"` the legend jumps to a new
    corner whenever a step adds a labelled artist, which erases its own text
    where it was and covers content where it lands. That is chrome moving, not
    a step deleting a construction.
    """
    legend = ax.get_legend()
    if legend is None:
        return None
    try:
        bbox = legend.get_window_extent()
        height = fig.get_window_extent().height
    except Exception:
        return None
    pad = 2
    return [
        int(round(bbox.x0)) - pad,
        int(round(bbox.x1)) + pad,
        int(round(height - bbox.y1)) - pad,
        int(round(height - bbox.y0)) + pad,
    ]


def _marked_points(ax):
    """Points the code explicitly MARKED.

    Scatter offsets (`PathCollection` only — a `fill_between` polygon or a
    `LineCollection` reports a phantom offset at the origin), vertices of lines
    that carry a marker, and the centres of circle/ellipse patches. Line
    vertices without a marker are deliberately excluded: a perpendicular
    bisector passes through the circumcenter, so counting its sampled vertices
    would let a solution "hit" the answer without ever drawing it.
    """
    from matplotlib.collections import PathCollection
    import numpy as np

    pts = []
    for coll in ax.collections:
        if not isinstance(coll, PathCollection):
            continue
        try:
            offsets = np.asarray(coll.get_offsets(), dtype=float)
        except Exception:
            continue
        if offsets.size:
            pts.extend(offsets.reshape(-1, 2).tolist())
    for line in ax.lines:
        if line.get_marker() in (None, "None", "", " "):
            continue
        data = np.asarray(line.get_xydata(), dtype=float)
        if data.size:
            pts.extend(data.reshape(-1, 2).tolist())
    for patch in ax.patches:
        centre = getattr(patch, "center", None)
        if centre is not None:
            try:
                pts.append([float(centre[0]), float(centre[1])])
            except (TypeError, ValueError, IndexError):
                pass
    return pts


def main() -> None:
    job = json.loads(Path(sys.argv[1]).read_text())
    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    ns: dict = {"__name__": "noregeo_canvas"}
    try:
        exec(compile(job["preamble"], "<preamble>", "exec"), ns)
    except Exception:
        json.dump(
            {"preamble_error": traceback.format_exc(), "steps": []},
            sys.stdout,
        )
        return

    fig, ax = ns["fig"], ns["ax"]
    base_path = out_dir / "canvas_base.png"
    fig.savefig(base_path, dpi=job["dpi"], facecolor="white")
    payload = {
        "preamble_error": None,
        "base_frame": {
            "image": str(base_path),
            "view_box": _view_box(ax),
            "axes_box": _axes_box_px(fig, ax),
            "legend_box": _legend_box_px(fig, ax),
        },
        "steps": [],
    }

    for i, step in enumerate(job["steps"], 1):
        rec = {"index": i, "ok": False, "image": None, "error": None}
        try:
            exec(compile(step["code"], f"<step {i}>", "exec"), ns)
            title = step.get("title") or ""
            ns["ax"].set_title(f"Step {i}. {title}", fontsize=11, wrap=True)
            path = out_dir / f"step_{i:02d}.png"
            # No bbox_inches="tight": every PNG must have identical pixel
            # dimensions so consecutive steps can be diffed.
            ns["fig"].savefig(path, dpi=job["dpi"], facecolor="white")
            rec["ok"] = True
            rec["image"] = str(path)
            rec["view_box"] = _view_box(ns["ax"])
            rec["axes_box"] = _axes_box_px(ns["fig"], ns["ax"])
            rec["legend_box"] = _legend_box_px(ns["fig"], ns["ax"])
        except Exception:
            rec["error"] = traceback.format_exc(limit=4)
        payload["steps"].append(rec)
        if not rec["ok"]:
            break

    if payload["steps"] and all(r["ok"] for r in payload["steps"]):
        state = ns.get("state", {})
        payload["state_answer"] = _jsonable(
            state.get("answer") if isinstance(state, dict) else None
        )
        payload["state_keys"] = (
            sorted(map(str, state)) if isinstance(state, dict) else []
        )
        target = job.get("target_point")
        if target is not None:
            import numpy as np

            pts = _marked_points(ns["ax"])
            if pts:
                arr = np.asarray(pts, dtype=float)
                d = np.hypot(arr[:, 0] - target[0], arr[:, 1] - target[1])
                payload["answer_point_distance"] = float(d.min())
            else:
                payload["answer_point_distance"] = None
            payload["marked_points"] = len(pts)

    json.dump(payload, sys.stdout)


if __name__ == "__main__":
    main()
