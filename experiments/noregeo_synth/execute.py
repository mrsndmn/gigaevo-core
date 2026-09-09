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


@dataclass
class ExecResult:
    ok: bool
    steps: list[StepResult]
    error: str | None = None  # preamble / runner-level failure
    # What the code itself claims the answer is (`state["answer"]`), and how far
    # the nearest marked point on the finished canvas is from the golden point.
    state_answer: object = None
    state_keys: list[str] = field(default_factory=list)
    answer_point_distance: float | None = None
    marked_points: int = 0

    @property
    def first_failure(self) -> StepResult | None:
        return next((s for s in self.steps if not s.ok), None)


def run_steps(
    preamble: str,
    steps: list[dict],
    out_dir: Path,
    cfg: GenConfig,
    target_point: tuple[float, float] | None = None,
) -> ExecResult:
    """Execute the steps in one isolated subprocess, saving a PNG per step."""
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

    out_dir.mkdir(parents=True, exist_ok=True)
    # A repair round can produce fewer steps than the round before it; without
    # this the directory would keep an orphaned step_05.png from the old try.
    for stale in out_dir.glob("step_*.png"):
        stale.unlink()
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
        return ExecResult(
            ok=False, steps=[], error=f"timeout after {cfg.exec_timeout}s"
        )
    finally:
        job_path.unlink(missing_ok=True)

    if proc.returncode != 0 or not proc.stdout.strip():
        return ExecResult(
            ok=False,
            steps=[],
            error=f"runner exited {proc.returncode}: {proc.stderr[-2000:]}",
        )

    payload = json.loads(proc.stdout)
    if payload.get("preamble_error"):
        return ExecResult(ok=False, steps=[], error=payload["preamble_error"])

    results = [
        StepResult(
            index=s["index"], ok=s["ok"], image=s.get("image"), error=s.get("error")
        )
        for s in payload["steps"]
    ]
    ok = len(results) == len(steps) and all(s.ok for s in results)
    return ExecResult(
        ok=ok,
        steps=results,
        state_answer=payload.get("state_answer"),
        state_keys=payload.get("state_keys", []),
        answer_point_distance=payload.get("answer_point_distance"),
        marked_points=payload.get("marked_points", 0),
    )
