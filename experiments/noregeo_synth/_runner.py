"""Subprocess entrypoint: execute preamble + step snippets, one PNG per step.

Run as ``python -I _runner.py job.json`` — `-I` drops PYTHONPATH and the user
site dir, so this file must stay self-contained (stdlib + matplotlib/numpy from
the active env). It prints one JSON object on stdout.

The whole point of a single process for the whole sequence is incrementality:
the namespace (and therefore the figure) persists across steps, so step k's PNG
contains everything steps 1..k-1 drew.

Besides the images it reports two things the caller uses to decide whether the
sample is honest: the value the code itself put in ``state["answer"]``, and how
far the nearest *marked* point on the finished canvas is from the benchmark's
golden point. A trace whose prose is right but whose drawing lands somewhere
else is worse than no trace at all.
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


def _marked_points(ax):
    """Points the code explicitly MARKED (scatter offsets + marked vertices).

    Line vertices without a marker are deliberately excluded: a perpendicular
    bisector passes through the circumcenter, so counting its sampled vertices
    would let a solution "hit" the answer without ever drawing it.
    """
    import numpy as np

    pts = []
    for coll in ax.collections:
        try:
            offsets = np.asarray(coll.get_offsets(), dtype=float)
        except Exception:
            continue
        if offsets.size:
            pts.extend(offsets.reshape(-1, 2).tolist())
    for line in ax.lines:
        marker = line.get_marker()
        if marker in (None, "None", "", " "):
            continue
        data = np.asarray(line.get_xydata(), dtype=float)
        if data.size:
            pts.extend(data.reshape(-1, 2).tolist())
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

    results = []
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
        except Exception:
            rec["error"] = traceback.format_exc(limit=4)
        results.append(rec)
        if not rec["ok"]:
            break

    payload = {"preamble_error": None, "steps": results}

    if results and all(r["ok"] for r in results):
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
