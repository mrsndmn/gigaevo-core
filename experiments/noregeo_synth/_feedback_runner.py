"""Persistent JSON-lines drawing worker, launched with python -I.

Only the parent supplies the trusted preamble. Each generated step has passed
its AST guard. This is process isolation, not a security sandbox.
"""

from contextlib import redirect_stdout
import importlib.util
import json
from pathlib import Path
import sys
import traceback

# Load only the known sibling helper module, without restoring PYTHONPATH.
spec = importlib.util.spec_from_file_location(
    "frame_helpers", Path(__file__).with_name("_runner.py")
)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


def frame(ns, path, dpi):
    fig, ax = ns["fig"], ns["ax"]
    fig.savefig(path, dpi=dpi, facecolor="white")
    return {
        "image": str(path),
        "view_box": helpers._view_box(ax),
        "axes_box": helpers._axes_box_px(fig, ax),
        "legend_box": helpers._legend_box_px(fig, ax),
    }


def main():
    ns = {"__name__": "noregeo_canvas"}
    job = None
    for line in sys.stdin:
        try:
            message = json.loads(line)
            # User print() cannot corrupt the wire protocol or block its stdout.
            with redirect_stdout(sys.stderr):
                if job is None:
                    job = message
                    exec(compile(job["preamble"], "<preamble>", "exec"), ns)
                    out = Path(job["out_dir"])
                    payload = {
                        "ok": True,
                        "base_frame": frame(ns, out / "canvas_base.png", job["dpi"]),
                    }
                else:
                    i = message["index"]
                    step = message["step"]
                    exec(compile(step["code"], f"<step {i}>", "exec"), ns)
                    ns["ax"].set_title(
                        f"Step {i}. {step['title']}", fontsize=11, wrap=True
                    )
                    step_result = {
                        "index": i,
                        "ok": True,
                        **frame(ns, out / f"step_{i:02d}.png", job["dpi"]),
                    }
                    state = ns.get("state", {})
                    payload = {
                        "ok": True,
                        "step": step_result,
                        "state_answer": helpers._jsonable(state.get("answer")),
                        "state_keys": sorted(map(str, state)),
                    }
                    target = job.get("target_point")
                    if target is not None:
                        import numpy as np

                        points = helpers._marked_points(ns["ax"])
                        payload["marked_points"] = len(points)
                        payload["answer_point_distance"] = (
                            float(
                                np.linalg.norm(
                                    np.asarray(points) - target, axis=1
                                ).min()
                            )
                            if points
                            else None
                        )
            print(json.dumps(payload), flush=True)
        except Exception:
            print(
                json.dumps({"ok": False, "error": traceback.format_exc(limit=4)}),
                flush=True,
            )
            return


if __name__ == "__main__":
    main()
