"""Run the generated step code and collect one PNG per reasoning step."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from .configs import GenConfig
from .guard import check

RUNNER = Path(__file__).with_name("_runner.py")


@dataclass
class StepResult:
    index: int
    ok: bool
    image: str | None = None
    error: str | None = None
    guard_problems: list[str] = field(default_factory=list)
    # The frame this step produced: data-space limits and the axes rectangle in
    # image pixels, both needed to compare it against the frame before it.
    view_box: list[float] | None = None
    axes_box: list[int] | None = None
    legend_box: list[int] | None = None

    @property
    def frame(self) -> dict:
        return {
            "image": self.image,
            "view_box": self.view_box,
            "axes_box": self.axes_box,
            "legend_box": self.legend_box,
        }


@dataclass
class ExecResult:
    ok: bool
    steps: list[StepResult]
    error: str | None = None  # preamble / runner-level failure
    # The canvas with the preamble alone: what `verify.compare_frames` diffs
    # step 1 against, so that "step 1 drew nothing" is detectable at all.
    base_frame: dict | None = None
    # What the code itself claims the answer is (`state["answer"]`), and how far
    # the nearest marked point on the finished canvas is from the golden point.
    state_answer: object = None
    state_keys: list[str] = field(default_factory=list)
    answer_point_distance: float | None = None
    marked_points: int = 0

    @property
    def first_failure(self) -> StepResult | None:
        return next((s for s in self.steps if not s.ok), None)


def _clear_frames(out_dir: Path) -> None:
    """Drop the PNGs of a previous attempt.

    A repair round can produce fewer steps than the round before it, and a
    rejected or timed-out round produces none at all; without this the sample
    directory would keep an orphaned step_05.png that no record refers to.
    """
    for stale in out_dir.glob("step_*.png"):
        stale.unlink()
    (out_dir / "canvas_base.png").unlink(missing_ok=True)


def run_steps(
    preamble: str,
    steps: list[dict],
    out_dir: Path,
    cfg: GenConfig,
    target_point: tuple[float, float] | None = None,
) -> ExecResult:
    """Execute the steps in one isolated subprocess, saving a PNG per step."""
    out_dir.mkdir(parents=True, exist_ok=True)
    _clear_frames(out_dir)

    blocked = {
        i: hits for i, s in enumerate(steps, 1) if (hits := check(s.get("code", "")))
    }
    if blocked:
        first = min(blocked)
        joined = "; ".join(blocked[first])
        return ExecResult(
            ok=False,
            steps=[
                StepResult(
                    index=first,
                    ok=False,
                    guard_problems=blocked[first],
                    error=f"rejected before execution: {joined}",
                )
            ],
            error=f"step {first} rejected by the static guard: {joined}",
        )

    job = {
        "preamble": preamble,
        "steps": [{"title": s.get("title", ""), "code": s["code"]} for s in steps],
        "out_dir": str(out_dir),
        "dpi": cfg.dpi,
        "target_point": list(target_point) if target_point is not None else None,
    }
    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False, dir=out_dir
    ) as fh:
        json.dump(job, fh)
        job_path = Path(fh.name)

    try:
        proc = subprocess.run(
            [sys.executable, "-I", str(RUNNER), str(job_path)],
            capture_output=True,
            text=True,
            timeout=cfg.exec_timeout,
        )
    except subprocess.TimeoutExpired:
        _clear_frames(out_dir)  # the killed child may have written part of them
        return ExecResult(
            ok=False, steps=[], error=f"timeout after {cfg.exec_timeout}s"
        )
    finally:
        job_path.unlink(missing_ok=True)

    if proc.returncode != 0 or not proc.stdout.strip():
        _clear_frames(out_dir)
        return ExecResult(
            ok=False,
            steps=[],
            error=f"runner exited {proc.returncode}: {proc.stderr[-2000:]}",
        )

    payload = json.loads(proc.stdout)
    if payload.get("preamble_error"):
        _clear_frames(out_dir)
        return ExecResult(ok=False, steps=[], error=payload["preamble_error"])

    results = [
        StepResult(
            index=s["index"],
            ok=s["ok"],
            image=s.get("image"),
            error=s.get("error"),
            view_box=s.get("view_box"),
            axes_box=s.get("axes_box"),
            legend_box=s.get("legend_box"),
        )
        for s in payload["steps"]
    ]
    ok = len(results) == len(steps) and all(s.ok for s in results)
    return ExecResult(
        ok=ok,
        steps=results,
        base_frame=payload.get("base_frame"),
        state_answer=payload.get("state_answer"),
        state_keys=payload.get("state_keys", []),
        answer_point_distance=payload.get("answer_point_distance"),
        marked_points=payload.get("marked_points", 0),
    )
