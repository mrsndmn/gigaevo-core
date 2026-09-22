"""Paired stepwise NoReGeo rollouts with or without rendered feedback.

Run: python -m experiments.noregeo_synth.feedback --task outer_circle_center:6200
Read existing results: add --summary --out <run directory>.
No training, gold feedback, execution repairs, or benchmark images are used.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
import json
from pathlib import Path
import time

from . import dataset as ds
from .canvas import build_preamble
from .configs import EXCLUDED_STEMS, REPO_ROOT, GenConfig
from .feedback_execute import DrawingSession
from .generate import (
    _call,
    _client,
    batch_stopper,
    check_model_is_served,
    golden_point,
    run_checks,
)
from .prompts import _answer_schema, build_messages
from .verify import contact_sheet


@dataclass(frozen=True)
class FeedbackExperiment:
    name: str
    observation: str
    generation: GenConfig


# The complete experiment design lives here. Equal action and token budgets;
# initial coordinates and all previous actions remain available in both arms.
_BASE = GenConfig(
    name="qwen38-feedback",
    min_steps=4,
    max_steps=4,
    max_tokens=3072,
    max_tokens_ceiling=3072,
    reveal_answer=False,
    reject_pasted_answer=False,
    max_repair_rounds=0,
)
experiment_configs = [
    FeedbackExperiment("history", "status", _BASE),
    FeedbackExperiment("canvas", "image", _BASE),
]
DEFAULT_TASK = "outer_circle_center:6200"
OUT_ROOT = REPO_ROOT / "run" / "noregeo_feedback"


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def step_schema():
    return {
        "name": "construction_step",
        "schema": {
            "type": "object",
            "properties": {
                k: {"type": "string"} for k in ("title", "reasoning", "code")
            },
            "required": ["title", "reasoning", "code"],
            "additionalProperties": False,
        },
    }


def answer_schema(row):
    return {
        "name": "construction_answer",
        "schema": {
            "type": "object",
            "properties": {"final_answer": _answer_schema(row)},
            "required": ["final_answer"],
            "additionalProperties": False,
        },
    }


def observation(image: Path, mode: str, instruction: str):
    """Only the image attachment differs between experimental arms."""
    content = [{"type": "text", "text": "Execution succeeded.\n" + instruction}]
    if mode == "image":
        encoded = base64.b64encode(image.read_bytes()).decode()
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64," + encoded,
                },
            }
        )
    elif mode != "status":
        raise ValueError(f"unknown observation mode: {mode}")
    return {"role": "user", "content": content}


def initial_messages(task, cfg, preamble):
    # Force these switches even for programmatic callers: gold is offline only.
    cfg = replace(cfg, reveal_answer=False)
    messages = build_messages(task.row, cfg, preamble)
    messages.append(
        {
            "role": "user",
            "content": (
                f"We will execute exactly {cfg.max_steps} construction steps interactively. "
                "Return ONLY the next step object (title, reasoning, code), not a full "
                "solution. You cannot edit earlier code. Use deterministic operations. "
                "Execution status and, when available, its rendered canvas follow each step. "
                "Use that feedback to choose the next action. On the last construction "
                "step compute, draw and store state['answer']. After its execution you "
                "will be asked separately for final_answer. Start with step 1 now."
            ),
        }
    )
    return messages


def render_report(record, out):
    lines = [f"# {record['task_id']} — {record['arm']}", "", record["question"], ""]
    for i, step in enumerate(record["steps"], 1):
        lines += [
            f"## Step {i}: {step['title']}",
            "",
            step["reasoning"],
            "",
            "```python",
            step["code"],
            "```",
            "",
        ]
        frame = out / "frames" / f"step_{i:02d}.png"
        if frame.exists():
            lines += [f"![Step {i}]({frame.relative_to(out)})", ""]
    lines += [
        "## Offline result",
        "",
        "```json",
        json.dumps(
            {
                k: record.get(k)
                for k in (
                    "final_answer",
                    "golden_answer",
                    "accepted",
                    "error",
                    "checks",
                )
            },
            ensure_ascii=False,
            indent=2,
        ),
        "```",
        "",
    ]
    (out / "trace.md").write_text("\n".join(lines))
    frames = [
        out / "frames" / f"step_{i:02d}.png" for i in range(1, len(record["steps"]) + 1)
    ]
    frames = [str(p) for p in frames if p.exists()]
    if frames:
        contact_sheet(
            frames,
            [f"Step {i}" for i in range(1, len(frames) + 1)],
            out / "contact_sheet.png",
            suptitle=record["arm"],
        )


def rollout(task, experiment, out, client, call=_call, first_step=None):
    """Keep one drawing process alive across all policy calls in this rollout."""
    cfg = experiment.generation
    if cfg.min_steps != cfg.max_steps or cfg.max_steps < 1:
        raise ValueError("feedback rollouts require a fixed positive step budget")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    preamble = build_preamble(task.row, cfg)
    (out / "preamble.py").write_text(preamble)
    messages = initial_messages(task, cfg, preamble)
    record = {
        "task_id": task.task_id,
        "question": task.row["question"],
        "arm": experiment.name,
        "config": asdict(experiment),
        "steps": [],
        "calls": [],
        "golden_answer": task.row.get("answer"),
        "accepted": False,
        "error": None,
        "status": "running",
        "shared_first_step": first_step is not None,
    }
    started = time.monotonic()
    session = None

    def checkpoint():
        record["wall_time_s"] = round(time.monotonic() - started, 2)
        write_json(out / "record.json", record)

    def request(schema, index):
        write_json(
            out / f"request_{index:02d}.json", {"messages": messages, "schema": schema}
        )
        payload, thinking, usage = call(
            client, cfg, messages, schema, salt=f"{task.task_id}:feedback:{index}"
        )
        record["calls"].append(
            {"index": index, "usage": usage, "thinking_chars": len(thinking or "")}
        )
        write_json(out / f"response_{index:02d}.json", payload)
        return payload

    checkpoint()
    try:
        session = DrawingSession(preamble, out / "frames", cfg, golden_point(task.row))
        for i in range(1, cfg.max_steps + 1):
            if i == 1 and first_step is not None:
                step = dict(first_step)
                write_json(out / "shared_first_step.json", step)
            else:
                step = request(step_schema(), i)
            if set(step) != {"title", "reasoning", "code"} or not all(
                isinstance(v, str) for v in step.values()
            ):
                raise ValueError("invalid step object")
            record["steps"].append(step)
            messages.append({"role": "assistant", "content": json.dumps(step)})
            result = session.step(step)
            write_json(out / f"execution_{i:02d}.json", asdict(result))
            if not result.ok:
                record["status"] = "execution_failed"
                failed = result.first_failure
                record["error"] = result.error or (
                    failed.error if failed else "execution failed"
                )
                return record
            instruction = (
                "All construction steps are complete. Return only final_answer now."
                if i == cfg.max_steps
                else f"Return construction step {i + 1} of {cfg.max_steps} now."
            )
            messages.append(
                observation(
                    Path(result.steps[-1].image), experiment.observation, instruction
                )
            )
            checkpoint()
            print(
                f"{task.task_id} {experiment.name}: executed step {i}/{cfg.max_steps}",
                flush=True,
            )
        answer = request(answer_schema(task.row), cfg.max_steps + 1)
        record["final_answer"] = answer.get("final_answer")
        record["checks"] = run_checks(
            task.row, record, result, replace(cfg, reject_pasted_answer=False)
        )
        record["accepted"] = not record["checks"]["problems"]
        record["status"] = "completed"
        return record
    except Exception as exc:
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if session is not None:
            session.close()
        checkpoint()
        render_report(record, out)


def summary(root):
    records = sorted(Path(root).glob("*/*/record.json"))
    if not records:
        raise ValueError(f"no feedback records in {root}")
    for path in records:
        r = json.loads(path.read_text())
        print(
            f"{r['task_id']} {r['arm']}: {r['status']}, accepted={r['accepted']}, "
            f"answer={r.get('final_answer')}, error={r.get('error')}"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", action="append", help="repeat for paired tasks")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args(argv)
    if args.summary:
        if not args.out:
            parser.error("--summary requires --out")
        summary(args.out)
        return 0
    pool = ds.load_tasks()
    tasks = [
        ds.select(pool, spec) for spec in dict.fromkeys(args.task or [DEFAULT_TASK])
    ]
    for task in tasks:
        if task.stem in EXCLUDED_STEMS:
            parser.error(f"excluded task {task.task_id}: {EXCLUDED_STEMS[task.stem]}")
    root = (
        args.out or OUT_ROOT / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    ).resolve()
    # Refuse overwrite before any inference, including partial existing runs.
    if root.exists():
        parser.error(f"output already exists: {root}; use --summary or a new --out")
    client, _ = _client(_BASE)
    check_model_is_served(client, _BASE)
    root.mkdir(parents=True)
    write_json(
        root / "manifest.json",
        {
            "experiments": [asdict(c) for c in experiment_configs],
            "tasks": [t.task_id for t in tasks],
        },
    )
    print(f"Artifacts: {root}", flush=True)
    failed = False
    for task in tasks:
        shared = None
        for experiment in experiment_configs:
            destination = root / task.task_id.replace(":", "_") / experiment.name
            try:
                record = rollout(
                    task, experiment, destination, client, first_step=shared
                )
            except Exception as exc:
                if batch_stopper(exc):
                    raise
                failed = True
                record_path = destination / "record.json"
                if not record_path.exists():
                    raise
                record = json.loads(record_path.read_text())
                print(
                    f"{task.task_id} {experiment.name}: {record['error']}", flush=True
                )
            # Both arms branch from exactly the same first proposed action.
            if shared is None and record["steps"]:
                shared = record["steps"][0]
            failed |= record["status"] != "completed"
    summary(root)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
