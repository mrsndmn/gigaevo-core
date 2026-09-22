"""One persistent guarded drawing subprocess per interactive rollout."""

import json
from pathlib import Path
import selectors
import subprocess
import sys

from .execute import ExecResult, StepResult
from .guard import check


class DrawingSession:
    def __init__(self, preamble, out, cfg, target):
        self.cfg = cfg
        out.mkdir(parents=True, exist_ok=True)
        self.log = (out / "worker.log").open("w")
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-I",
                str(Path(__file__).with_name("_feedback_runner.py")),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            bufsize=1,
        )
        self.result = ExecResult(ok=True, steps=[])
        try:
            payload = self.exchange(
                {
                    "preamble": preamble,
                    "out_dir": str(out),
                    "dpi": cfg.dpi,
                    "target_point": target,
                }
            )
            if not payload["ok"]:
                raise RuntimeError(payload["error"])
            self.result.base_frame = payload["base_frame"]
        except BaseException:
            self.close()
            raise

    def exchange(self, value):
        self.process.stdin.write(json.dumps(value) + "\n")
        self.process.stdin.flush()
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(self.cfg.exec_timeout):
                self.close()
                raise TimeoutError(f"drawing step exceeded {self.cfg.exec_timeout}s")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("drawing worker exited; see frames/worker.log")
        return json.loads(line)

    def step(self, step):
        index = len(self.result.steps) + 1
        problems = check(step["code"])
        if problems:
            payload = {"ok": False, "error": "; ".join(problems)}
        else:
            payload = self.exchange({"index": index, "step": step})
        if not payload["ok"]:
            self.result.ok = False
            self.result.error = payload["error"]
            self.result.steps.append(
                StepResult(
                    index=index,
                    ok=False,
                    error=payload["error"],
                    guard_problems=problems,
                )
            )
        else:
            self.result.steps.append(StepResult(**payload["step"]))
            for key in (
                "state_answer",
                "state_keys",
                "answer_point_distance",
                "marked_points",
            ):
                if key in payload:
                    setattr(self.result, key, payload[key])
        return self.result

    def close(self):
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait()
        self.process.stdin.close()
        self.process.stdout.close()
        self.log.close()
