"""Staged offline-format diagnosis and paired recoverable canvas evaluation.

All inference parameters and advance gates live in experiment_configs. Old
artifacts are read only. --prepare freezes inputs; --run executes; --report
renders a standalone HTML report without inference or generated-code execution.
"""

from __future__ import annotations

import argparse
import ast
import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from html import escape
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from .canvas import build_preamble
from .dataset import Task
from .feedback import _BASE, answer_schema, initial_messages, step_schema
from .feedback_eval import SPEC as OLD_SPEC
from .feedback_eval import atomic_text, digest, paired_counts, read, write
from .generate import _client as _base_client
from .generate import _seed_for, check_model_is_served
from .guard import check
from .verify import answer_matches, compare_frames


@dataclass(frozen=True)
class RecoveryExperiment:
    name: str = "qwen38-recovery-v2"
    source: str = str(Path(OLD_SPEC.artifact_root) / OLD_SPEC.name)
    artifact_root: str = OLD_SPEC.artifact_root
    instance_type: str = "a100.1gpu"
    workers: int = 2
    diagnostic_count: int = 25
    diagnostic_min_valid: int = 23
    pilot_count: int = 15
    pilot_min_completed: int = 27
    max_actions: int = 4
    repair_budget: int = 2
    max_transport_retries: int = 3
    output_chars: int = 4096
    # Technical gate only: no threshold on answer correctness.
    pilot_max_format_failure_rate: float = 0.1
    injection_offset: tuple[float, float] = (2.0, 0.0)


experiment_configs = [RecoveryExperiment()]
SPEC = experiment_configs[0]
CFG = replace(_BASE, call_retries=0)
FORMATS = ("json", "block")
ARMS = ("history", "canvas")
ENV_HELP = """
# 2D determinant: unlike this worker's np.cross, this accepts length-two vectors.
def cross2d(a, b):
    return float(a[0] * b[1] - a[1] * b[0])
"""


def _client(cfg):
    client, credentials = _base_client(cfg)
    return client.with_options(max_retries=0), credentials


def policy_task(task):
    # Allowlist input fields: never let answer-derived view bounds enter prompts.
    fields = ("question", "points", "schema_type", "choices", "qid")
    return replace(task, row={key: task.row[key] for key in fields if key in task.row})


def prepare(out, spec=SPEC):
    source = Path(spec.source)
    frozen = read(source / "inputs.json")
    cases = []
    for p in sorted(source.glob("tasks/*/rollouts/sample/*/record.json")):
        record = read(p)
        if "syntax error" not in (record.get("error") or ""):
            continue
        index = len(record["steps"])
        request = p.parent / f"request_{index:02d}.json"
        cases.append(
            {
                "id": record["task_id"].replace(":", "_") + "_" + record["arm"],
                "request": read(request),
                "source": str(p.relative_to(source)),
                "source_sha256": digest(p),
                "request_sha256": digest(request),
            }
        )
    if len(cases) != spec.diagnostic_count:
        raise ValueError("diagnostic cohort changed")
    tasks = sorted(frozen["tasks"], key=lambda t: (t["stem"], t["row"]["qid"]))
    if len(tasks) != 75:
        raise ValueError("historical cohort changed")
    # One per stem, evenly spread over alphabetically sorted stems, outcome blind.
    first = {t["stem"]: t for t in reversed(tasks)}
    stems = sorted(first)
    indices = [
        i * (len(stems) - 1) // (spec.pilot_count - 1) for i in range(spec.pilot_count)
    ]
    pilot = [first[stems[i]] for i in indices]
    value = {
        "design": asdict(spec),
        "generation": asdict(CFG),
        "cases": cases,
        "tasks": tasks,
        "pilot": pilot,
        "injection": [t for t in tasks if t["stem"] == "center_coordinates"],
        "source_inputs_sha256": digest(source / "inputs.json"),
    }
    value = json.loads(json.dumps(value))
    path = Path(out) / "inputs.json"
    if path.exists() and read(path) != value:
        raise ValueError("frozen inputs/design mismatch; use a new experiment name")
    write(path, value)
    return value


def request(client, messages, schema, fmt, directory, salt, spec=SPEC):
    """No hidden completion re-rolls. Every raw response/transport error saved."""
    directory = Path(directory)
    cached = directory / "result.json"
    if cached.exists():
        return read(cached)
    directory.mkdir(parents=True, exist_ok=True)
    if (
        (directory / "request.json").exists()
        or list(directory.glob("raw_*.json"))
        or list(directory.glob("error_*.json"))
    ):
        raise RuntimeError(f"incomplete request at {directory}; refusing silent reroll")
    kwargs = dict(
        model=CFG.model,
        messages=messages,
        temperature=CFG.temperature,
        top_p=CFG.top_p,
        max_tokens=CFG.max_tokens,
        seed=_seed_for(CFG, salt, 0),
        extra_body={"reasoning_effort": CFG.thinking},
    )
    if fmt == "json":
        kwargs["response_format"] = {"type": "json_schema", "json_schema": schema}
    write(directory / "request.json", kwargs)
    started = time.monotonic()
    for attempt in range(spec.max_transport_retries + 1):
        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as exc:
            write(
                directory / f"error_{attempt:02d}.json",
                {"type": type(exc).__name__, "error": str(exc)},
            )
            # Only retry transient transport errors, not auth/config failures.
            from .generate import _is_retryable

            if _is_retryable(exc) and attempt < spec.max_transport_retries:
                time.sleep(min(2 ** (attempt + 1), 8))
                continue
            result = {
                "error": f"{type(exc).__name__}: {exc}",
                "content": None,
                "wall_time_s": time.monotonic() - started,
            }
            write(cached, result)
            return result
        write(directory / f"raw_{attempt:02d}.json", response.model_dump())
        choice = response.choices[0]
        result = {
            "content": choice.message.content or "",
            "error": None,
            "finish_reason": choice.finish_reason,
            "usage": response.usage.model_dump() if response.usage else {},
            "wall_time_s": time.monotonic() - started,
        }
        write(cached, result)
        return result


def parse_step(content, fmt, interactive=False):
    if fmt == "json":
        value = json.loads(content)
        keys = {"title", "reasoning", "code"} | ({"action"} if interactive else set())
        if not isinstance(value, dict) or set(value) != keys:
            raise ValueError("incorrect step keys")
        if not all(isinstance(v, str) for v in value.values()):
            raise ValueError("step fields must be strings")
    else:
        # Exactly one fenced code block; separate metadata avoids escaping Python.
        match = re.fullmatch(r"\s*(.*?)\n```python\s*\n(.*?)\n```\s*", content, re.S)
        if not match:
            raise ValueError("expected metadata followed by one python code block")
        header, code = match.groups()
        lines = header.splitlines()
        if interactive:
            if not lines or not lines[0].startswith("ACTION: "):
                raise ValueError("missing ACTION header")
            action = lines.pop(0)[8:].strip()
        if not lines or not lines[0].startswith("TITLE: "):
            raise ValueError("missing TITLE header")
        title = lines.pop(0)[7:].strip()
        reasoning = "\n".join(lines)
        if not reasoning.startswith("REASONING: "):
            raise ValueError("missing REASONING header")
        value = {"title": title, "reasoning": reasoning[11:], "code": code}
        if interactive:
            value["action"] = action
    if interactive and value["action"] not in {"append", "replace_last", "finish"}:
        raise ValueError("unknown action")
    if not value["code"].strip():
        raise ValueError("empty code; use pass for finish")
    ast.parse(value["code"])
    return value


def format_instruction(fmt, interactive=False):
    if fmt == "json":
        return "Return only the JSON object matching the response schema."
    return (
        "Return exactly this format, without JSON or any additional blocks:\n"
        + (
            "ACTION: append | replace_last | finish (choose one)\n"
            if interactive
            else ""
        )
        + "TITLE: short title\nREASONING: explanation\n```python\n"
        "# complete Python code, with normal unescaped quotes\n```"
    )


def diagnose_one(case, fmt, out, spec=SPEC):
    path = Path(out) / "diagnostic" / case["id"] / fmt
    client, _ = _client(CFG)
    messages = list(case["request"]["messages"])
    if fmt == "block":
        messages.append({"role": "user", "content": format_instruction(fmt)})
    raw = request(
        client,
        messages,
        case["request"]["schema"],
        fmt,
        path,
        "format:" + case["id"],
        spec,
    )
    result = dict(raw)
    try:
        if raw["error"]:
            raise ValueError(raw["error"])
        step = parse_step(raw["content"], fmt)
        result.update(valid=True, step=step, guard_problems=check(step["code"]))
    except (ValueError, SyntaxError, TypeError) as exc:
        result.update(valid=False, parse_error=str(exc))
    write(path / "diagnostic.json", result)
    return result


def interactive_schema():
    schema = step_schema()
    schema["schema"]["properties"]["action"] = {
        "type": "string",
        "enum": ["append", "replace_last", "finish"],
    }
    schema["schema"]["required"].append("action")
    return schema


def preamble(task):
    return build_preamble(policy_task(task).row, CFG) + ENV_HELP


def messages_for(task, fmt):
    messages = initial_messages(policy_task(task), CFG, preamble(task))
    # Replace old fixed-step protocol, retain geometry/code environment contract.
    messages[-1] = {
        "role": "user",
        "content": (
            "This protocol replaces the fixed-step and strictly additive rules above. "
            f"You may commit up to {SPEC.max_actions} actions and repair up to "
            f"{SPEC.repair_budget} failed attempts. Finish as soon as solved. "
            "Choose append to add a step, replace_last to replace your last committed "
            "step (its effects are rolled back before execution), or finish. "
            "For finish use code pass; final_answer is requested separately. "
            "Errors roll back all effects of the failed attempt; you can retry. "
            "Execution stdout and state are returned in both modes; a canvas may also "
            "be attached. Use cross2d(a,b) for 2D cross products, not np.cross. "
            "Use plain text annotations, no dollar-delimited math or LaTeX. "
            "Compute and draw the result and assign state['answer'] before finish. "
            "Use deterministic code. Do not clear the figure. Start with append.\n"
            + format_instruction(fmt, True)
        ),
    }
    return messages


def execute(task, steps, directory):
    """Replay accepted actions in a fresh process: failed attempts cannot mutate state."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    problems = [p for step in steps for p in check(step["code"])]
    if problems:
        return {"ok": False, "error": "\n".join(problems), "stage": "guard"}
    job = {
        "preamble": preamble(task),
        "steps": steps,
        "dpi": CFG.dpi,
        "output_chars": SPEC.output_chars,
        "image": str((directory / "frame.png").resolve()),
    }
    write(directory / "job.json", job)
    try:
        with (directory / "stderr.log").open("w") as log:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(Path(__file__).with_name("_recovery_runner.py")),
                    str(directory / "job.json"),
                ],
                stdout=subprocess.PIPE,
                stderr=log,
                text=True,
                timeout=CFG.exec_timeout,
            )
        result = json.loads(proc.stdout)
    except subprocess.TimeoutExpired:
        result = {"ok": False, "error": "execution timeout", "stage": "execution"}
    except (ValueError, OSError) as exc:
        result = {"ok": False, "error": str(exc), "stage": "runner"}
    if result.get("ok"):
        delta = compare_frames([result["frame"]], baseline=result["baseline"])[0][
            "delta"
        ]
        result["visible"] = bool(
            result["added_artists"] > 0
            and delta is not None
            and delta >= CFG.min_pixel_delta
        )
    write(directory / "execution.json", result)
    return result


def feedback(result, arm, instruction):
    # Gold and scoring fields never enter this message.
    text = {k: result[k] for k in ("ok", "error", "stdout", "state") if k in result}
    content = [{"type": "text", "text": json.dumps(text) + "\n" + instruction}]
    if arm in {"canvas", "canvas_last"} and result.get("ok") and result.get("image"):
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64,"
                    + base64.b64encode(Path(result["image"]).read_bytes()).decode()
                },
            }
        )
    return {"role": "user", "content": content}


def context_for(messages, arm):
    """Keep all text; canvas_last transmits only the newest successful frame.

    Apply at every request boundary, including repair and final-answer calls.
    Do not mutate the canonical trace or already saved request snapshots.
    """
    if arm not in {"history", "canvas", "canvas_last"}:
        raise ValueError(f"unknown observation arm: {arm}")
    if arm != "canvas_last":
        return messages
    latest = None
    for i, message in enumerate(messages):
        if isinstance(message.get("content"), list):
            for j, item in enumerate(message["content"]):
                if item.get("type") == "image_url":
                    latest = (i, j)
    result = []
    for i, message in enumerate(messages):
        content = message.get("content")
        if isinstance(content, list):
            content = [
                item
                for j, item in enumerate(content)
                if item.get("type") != "image_url" or (i, j) == latest
            ]
            result.append({**message, "content": content})
        else:
            result.append(dict(message))
    return result


def rollout(task, fmt, arm, directory, first, injected=None, request_fn=None):
    request_call = request_fn or request
    directory = Path(directory)
    record_path = directory / "record.json"
    if record_path.exists():
        saved = read(record_path)
        if saved["status"] == "running":
            raise RuntimeError("interrupted rollout; refusing automatic reroll")
        return saved
    client = _client(CFG)[0] if request_fn is None else None
    messages = messages_for(task, fmt)
    accepted = []
    record = {
        "task_id": task.task_id,
        "question": task.row["question"],
        "arm": arm,
        "format": fmt,
        "status": "running",
        "events": [],
        "correct": False,
        "first": first,
        "injected": injected,
        "gold": task.row["answer"],
    }
    write(record_path, record)
    failures = 0
    committed = 0
    calls = 0
    result = None
    try:
        while committed < SPEC.max_actions and failures <= SPEC.repair_budget:
            calls += 1
            if calls == 1:
                raw = first
            else:
                raw = request_call(
                    client,
                    context_for(messages, arm),
                    interactive_schema(),
                    fmt,
                    directory / f"call_{calls:02d}",
                    f"{task.task_id}:recovery:{calls}",
                )
            event = {"call": calls, "raw": raw}
            record["events"].append(event)
            if raw.get("error"):
                record["status"] = "request_error"
                break
            messages.append({"role": "assistant", "content": raw["content"]})
            try:
                action = parse_step(raw["content"], fmt, True)
                if calls == 1 and action["action"] != "append":
                    raise ValueError("first action must append")
                event["action"] = action
                if action["action"] == "finish":
                    if not accepted or not construction_ready(result):
                        raise ValueError(
                            "Before finish, draw a visible construction and assign non-null state['answer']."
                        )
                    break
                if action["action"] == "replace_last" and not accepted:
                    raise ValueError("nothing to replace")
                prefix = (
                    accepted[:-1] if action["action"] == "replace_last" else accepted
                )
                trial = prefix + [action]
                outcome = execute(task, trial, directory / f"execution_{calls:02d}")
                event["execution"] = outcome
            except (ValueError, SyntaxError, TypeError) as exc:
                outcome = {"ok": False, "error": str(exc), "stage": "format"}
                event["execution"] = outcome
            if outcome["ok"]:
                accepted = trial
                result = outcome
                committed += 1
                # Controlled corruption is identical and visible as code in both arms.
                if injected is not None and committed == 1:
                    accepted.append(injected)
                    result = execute(task, accepted, directory / "injection")
                    if not result["ok"]:
                        raise RuntimeError("trusted injection failed")
                    event["injection"] = result
                    messages.append(feedback(outcome, arm, "Initial action committed."))
                    messages.append(
                        {
                            "role": "user",
                            "content": "A draft construction has now been committed. "
                            "It may contain an error; verify it. Its code:\n"
                            + injected["code"],
                        }
                    )
                messages.append(
                    feedback(
                        result,
                        arm,
                        f"{SPEC.max_actions - committed} action slots remain. "
                        "Continue, replace_last, or finish.",
                    )
                )
            else:
                failures += 1
                messages.append(
                    feedback(
                        outcome,
                        arm,
                        "Failed attempt rolled back; earlier committed state is unchanged. "
                        f"Repair slots remaining: {max(0, SPEC.repair_budget - failures + 1)}.",
                    )
                )
            write(record_path, record)
        record["failures"] = failures
        record["committed"] = committed
        record["steps"] = accepted
        if record["status"] == "request_error":
            return record
        if failures > SPEC.repair_budget:
            record["status"] = "execution_failed"
            return record
        record["construction_ready"] = construction_ready(result)
        if not record["construction_ready"]:
            record["status"] = "construction_failed"
            return record
        messages.append(
            {
                "role": "user",
                "content": "Action phase is over. Return only final_answer in the requested JSON schema.",
            }
        )
        final = request_call(
            client,
            context_for(messages, arm),
            answer_schema(task.row),
            "json",
            directory / "answer",
            task.task_id + ":final",
        )
        record["final_raw"] = final
        if final["error"]:
            record["status"] = "request_error"
        else:
            try:
                record["final_answer"] = json.loads(final["content"])["final_answer"]
                record["correct"], record["answer_detail"] = answer_matches(
                    task.row, record["final_answer"], CFG
                )
                record["code_answer_ok"], _ = answer_matches(
                    task.row, (result or {}).get("state", {}).get("answer"), CFG
                )
                record["status"] = "completed"
            except (ValueError, KeyError, TypeError) as exc:
                record.update(status="format_error", error=str(exc))
        return record
    except Exception as exc:
        record.update(
            status="infrastructure_error", error=f"{type(exc).__name__}: {exc}"
        )
        raise
    finally:
        write(record_path, record)


def evaluate_pair(value, fmt, out, injection=False):
    task = Task(**value)
    directory = (
        Path(out)
        / ("injection" if injection else "tasks")
        / task.task_id.replace(":", "_")
    )
    client, _ = _client(CFG)
    first = request(
        client,
        messages_for(task, fmt),
        interactive_schema(),
        fmt,
        directory / "first",
        task.task_id + ":recovery:1",
    )
    injected = None
    if injection:
        dx, dy = SPEC.injection_offset
        injected = {
            "action": "append",
            "title": "Draft midpoint",
            "reasoning": "A draft midpoint to verify.",
            "code": f'M = (P["A"] + P["B"]) / 2 + np.array([{dx}, {dy}])\n'
            'mark("M", M, color="tab:red")\nstate["answer"] = M',
        }
    return [rollout(task, fmt, arm, directory / arm, first, injected) for arm in ARMS]


def construction_ready(result):
    return bool(
        result
        and result.get("ok")
        and result.get("visible")
        and result.get("state", {}).get("answer") is not None
    )


def pilot_gate(records, spec=SPEC):
    events = [e for r in records for e in r["events"]]
    malformed = sum(e.get("execution", {}).get("stage") == "format" for e in events)
    completed = sum(
        r["status"] == "completed" and r.get("construction_ready", False)
        for r in records
    )
    rate = malformed / max(1, len(events))
    return {
        "pass": completed >= spec.pilot_min_completed
        and rate <= spec.pilot_max_format_failure_rate,
        "completed": completed,
        "planned": 2 * spec.pilot_count,
        "format_failure_rate": rate,
    }


def arm_metrics(rows):
    raws = [e["raw"] for r in rows for e in r["events"]]
    raws += [r["final_raw"] for r in rows if "final_raw" in r]
    return {
        "records": len(rows),
        "correct": sum(r["correct"] for r in rows),
        "completed": sum(r["status"] == "completed" for r in rows),
        "failed_actions": sum(r.get("failures", 0) for r in rows),
        "completed_after_failure": sum(
            r.get("failures", 0) > 0 and r["status"] == "completed" for r in rows
        ),
        "committed_actions": sum(r.get("committed", 0) for r in rows),
        "total_tokens_including_shared_first": sum(
            raw.get("usage", {}).get("total_tokens", 0) for raw in raws
        ),
        "request_seconds_including_shared_first": round(
            sum(raw.get("wall_time_s", 0) for raw in raws), 2
        ),
    }


def local_image(path):
    from experiments.vocab_factorization.jobs import to_local_path

    direct = Path(path)
    return direct if direct.exists() else Path(to_local_path(path))


def report(out):
    out = Path(out)
    snapshot = read(out / "inputs.json")
    parts = [
        '<!doctype html><html lang="ru"><meta charset="utf-8"><title>NoReGeo recovery</title>',
        "<style>body{font:16px/1.5 system-ui;margin:30px auto;max-width:1100px;padding:16px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eee;padding:12px}details{border:1px solid #ccc;padding:10px;margin:12px 0}img{max-width:500px;width:100%}td,th{padding:8px;border-bottom:1px solid #ccc}</style>",
        "<h1>NoReGeo: формат и восстановление после ошибок</h1>",
        "<p>Новый протокол; старые измерения не изменяются. "
        "Переходы между этапами зависят только от технических сбоев, не от правильности ответов.</p>",
    ]
    for name in ("format_gate", "pilot_gate", "complete", "stopped"):
        if (out / f"{name}.json").exists():
            parts.append(
                f"<h2>{name}</h2><pre>{escape(json.dumps(read(out / (name + '.json')), ensure_ascii=False, indent=2))}</pre>"
            )
    parts.append(
        "<h2>Диагностика формата</h2><table><tr><th>Формат</th><th>Проверено</th><th>Синтаксис верен</th><th>План</th></tr>"
    )
    for fmt in FORMATS:
        rows = [read(p) for p in out.glob(f"diagnostic/*/{fmt}/diagnostic.json")]
        parts.append(
            f"<tr><td>{fmt}</td><td>{len(rows)}</td><td>{sum(r['valid'] for r in rows)}</td><td>{len(snapshot['cases'])}</td></tr>"
        )
    parts.append("</table>")
    metrics = {}
    for phase, planned in [
        ("tasks", len(snapshot["tasks"])),
        ("injection", len(snapshot["injection"])),
    ]:
        parts.append(
            f"<h2>{phase}</h2><p>Пилот входит в основной набор; искусственные ошибки считаются отдельно.</p><table><tr><th>Вариант</th><th>Записей</th><th>Правильно</th><th>Завершено</th><th>План</th></tr>"
        )
        metrics[phase] = {}
        for arm in ARMS:
            rows = [read(p) for p in out.glob(f"{phase}/*/{arm}/record.json")]
            metrics[phase][arm] = arm_metrics(rows)
            parts.append(
                f"<tr><td>{arm}</td><td>{len(rows)}</td><td>{sum(r['correct'] for r in rows)}</td><td>{sum(r['status'] == 'completed' for r in rows)}</td><td>{planned}</td></tr>"
            )
        parts.append("</table>")
        for arm in ARMS:
            m = metrics[phase][arm]
            parts.append(
                f"<p>{arm}: ошибок действий {m['failed_actions']}, завершений после ошибки {m['completed_after_failure']}; "
                f"принятых действий {m['committed_actions']}; токенов {m['total_tokens_including_shared_first']}; "
                f"время запросов {m['request_seconds_including_shared_first']} с. Общий первый запрос включён в каждый вариант; суммы вариантов не являются общей стоимостью.</p>"
            )
        pairs = []
        for directory in sorted((out / phase).glob("*")):
            paths = [directory / arm / "record.json" for arm in ARMS]
            if all(p.exists() for p in paths):
                pair = [read(p) for p in paths]
                if all(
                    r["status"]
                    not in {"running", "request_error", "infrastructure_error"}
                    for r in pair
                ):
                    pairs.append(
                        tuple({"checks": {"answer_ok": r["correct"]}} for r in pair)
                    )
        metrics[phase]["paired"] = paired_counts(pairs)
        pair = metrics[phase]["paired"]
        parts.append(
            f"<p>Пар для сравнения: {pair['n']}; только history: {pair['history_only']}; только canvas: {pair['canvas_only']}; McNemar p={pair['mcnemar_exact_p']:.4g}. Незавершённые запросы исключены из пар, но остаются в общем плане.</p>"
        )
        for p in sorted(out.glob(f"{phase}/*/*/record.json")):
            r = read(p)
            parts.append(
                f"<details><summary>{escape(r['task_id'])} / {r['arm']}: {r['status']}, correct={r['correct']}</summary><p>{escape(r['question'])}</p>"
            )
            for event in r["events"]:
                parts.append(
                    f"<pre>{escape(event['raw'].get('content') or str(event['raw'].get('error')))}</pre>"
                )
                ex = event.get("execution", {})
                parts.append(
                    f"<pre>{escape(json.dumps({k: v for k, v in ex.items() if k != 'image'}, ensure_ascii=False))}</pre>"
                )
                for frame in (ex.get("image"), event.get("injection", {}).get("image")):
                    if frame and local_image(frame).exists():
                        parts.append(
                            '<img src="data:image/png;base64,'
                            + base64.b64encode(local_image(frame).read_bytes()).decode()
                            + '">'
                        )
            parts.append(
                f"<p>Ответ: {escape(str(r.get('final_answer')))}; эталон: {escape(str(r['gold']))}</p></details>"
            )
    for p in sorted(out.glob("diagnostic/*/*/diagnostic.json")):
        r = read(p)
        parts.append(
            f"<details><summary>{p.parent.parent.name} / {p.parent.name}: valid={r['valid']}</summary><pre>{escape(r.get('content') or str(r.get('error')))}</pre><p>{escape(r.get('parse_error', ''))}</p></details>"
        )
    parts.append("</html>")
    write(out / "metrics.json", metrics)
    atomic_text(out / "analysis.html", "\n".join(parts))


def run(out):
    out = Path(out)
    snapshot = read(out / "inputs.json")
    if snapshot["design"] != json.loads(json.dumps(asdict(SPEC))) or snapshot[
        "generation"
    ] != asdict(CFG):
        raise ValueError("worker design differs from frozen inputs")
    if (out / "complete.json").exists() or (out / "stopped.json").exists():
        report(out)
        return
    client, _ = _client(CFG)
    check_model_is_served(client, CFG)
    with ThreadPoolExecutor(max_workers=SPEC.workers) as pool:
        jobs = [(case, fmt) for case in snapshot["cases"] for fmt in FORMATS]
        list(pool.map(lambda pair: diagnose_one(*pair, out), jobs))
    scores = {
        fmt: sum(
            read(out / "diagnostic" / c["id"] / fmt / "diagnostic.json")["valid"]
            for c in snapshot["cases"]
        )
        for fmt in FORMATS
    }
    # Prefer block on a tie; this criterion is fixed before any calls.
    selected = max(FORMATS, key=lambda fmt: (scores[fmt], fmt == "block"))
    gate = {
        "scores": scores,
        "selected": selected,
        "pass": scores[selected] >= SPEC.diagnostic_min_valid,
    }
    write(out / "format_gate.json", gate)
    report(out)
    if not gate["pass"]:
        write(
            out / "stopped.json", {"reason": "format gate failed; pilot not launched"}
        )
        report(out)
        return
    with ThreadPoolExecutor(max_workers=SPEC.workers) as pool:
        results = list(
            pool.map(lambda t: evaluate_pair(t, selected, out), snapshot["pilot"])
        )
    gate = pilot_gate([r for pair in results for r in pair])
    write(out / "pilot_gate.json", gate)
    report(out)
    if not gate["pass"]:
        write(
            out / "stopped.json",
            {"reason": "technical pilot gate failed; full run not launched"},
        )
        report(out)
        return
    with ThreadPoolExecutor(max_workers=SPEC.workers) as pool:
        for _ in pool.map(lambda t: evaluate_pair(t, selected, out), snapshot["tasks"]):
            report(out)
        for _ in pool.map(
            lambda t: evaluate_pair(t, selected, out, True), snapshot["injection"]
        ):
            report(out)
    write(
        out / "complete.json",
        {
            "status": "all planned measurements attempted",
            "note": "request/format failures remain explicit; no rerolls",
        },
    )
    report(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare", "run", "report"):
        mode.add_argument("--" + name, action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.prepare:
        prepare(args.out)
    elif args.report:
        report(args.out)
    else:
        run(args.out)


if __name__ == "__main__":
    main()
