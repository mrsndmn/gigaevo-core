"""Isolated transactional replay worker. No benchmark gold enters this process."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import traceback

spec = importlib.util.spec_from_file_location(
    "helpers", Path(__file__).with_name("_runner.py")
)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


class BoundedOutput(io.StringIO):
    def __init__(self, limit):
        super().__init__()
        self.limit = limit

    def write(self, value):
        super().write(value[: max(0, self.limit - self.tell())])
        return len(value)


def main():
    job = json.loads(Path(sys.argv[1]).read_text())
    ns = {"__name__": "canvas_recovery"}
    output = BoundedOutput(job["output_chars"])
    index = -1
    try:
        with redirect_stdout(output):
            exec(compile(job["preamble"], "<preamble>", "exec"), ns)
            for index, step in enumerate(job["steps"]):
                output.seek(0)
                output.truncate()
                exec(compile(step["code"], f"<action {index + 1}>", "exec"), ns)
            ns["fig"].savefig(job["image"], dpi=job["dpi"], facecolor="white")
            result = {
                "ok": True,
                "image": job["image"],
                "stdout": output.getvalue(),
                "state": helpers._jsonable(ns.get("state", {})),
            }
    except Exception:
        result = {
            "ok": False,
            "error": traceback.format_exc(limit=5),
            "stage": "execution",
            "index": index,
            "stdout": output.getvalue(),
        }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
