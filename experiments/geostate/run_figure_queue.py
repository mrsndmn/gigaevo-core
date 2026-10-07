"""Run the paired figure cohorts sequentially in a local tmux session.

    python -u -m experiments.geostate.run_figure_queue

Queue order and all experiment settings live in configs.py. Each invocation
does one pass, recovering failed/interrupted requests first. Successful
results are cached. An unsuccessful cohort does not block later cohorts.
"""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
import time

from .client import write_json
from .configs import ARTIFACT_ROOT, FIGURE_QUEUE, config_named
from .consistency import COPY_DIR_RE, protocol_hash, recover_errors
from .run_figure_short50 import execute, freeze


def run_queue(root: Path) -> int:
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "figure-queue-state.json"
    with (root / ".figure-queue.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {"pid": os.getpid(), "started": time.time(), "status": "running",
                 "cohorts": {name: {"status": "pending"} for name in FIGURE_QUEUE}}
        write_json(state_path, state)
        failed = False
        for name in FIGURE_QUEUE:
            cfg = config_named(name)
            out = root / f"{name}-{protocol_hash(cfg)}"
            entry = state["cohorts"][name]
            entry.update(status="running", artifact_root=str(out), started=time.time(), workers=cfg.workers)
            write_json(state_path, state)
            print(f"QUEUE START {name}, workers={cfg.workers}, out={out}", flush=True)
            try:
                freeze(out, name)
                archived = recover_errors(out)
                print(f"QUEUE {name}: archived {len(archived)} failed/interrupted requests", flush=True)
                code = execute(name, out)
                entry.update(exit_code=code, status="complete" if code == 0 else "needs_retry")
                failed = failed or code != 0
            except Exception as exc:
                failed = True
                entry.update(status="error", error=f"{type(exc).__name__}: {exc}")
                print(f"QUEUE ERROR {name}: {entry['error']}", flush=True)
            entry.update(finished=time.time(), evaluated=sum(
                bool(COPY_DIR_RE.fullmatch(p.parent.name)) for p in out.glob("*/*/*/eval.json")
            ))
            write_json(state_path, state)
            print(f"QUEUE END {name}: {entry['status']}", flush=True)
        state.update(status="finished_with_errors" if failed else "complete", finished=time.time())
        write_json(state_path, state)
        return 1 if failed else 0


def main() -> int:
    return run_queue(Path(ARTIFACT_ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
