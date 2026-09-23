"""Full NoReGeo three-arm evaluation with durable checkpoints and failure browser."""

from __future__ import annotations

import argparse
import base64
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import fcntl
import hashlib
from html import escape
from itertools import permutations
import json
from pathlib import Path
import threading
import time

from . import dataset as ds
from . import recovery_eval as base
from . import recovery_extra as extra
from . import recovery_last as last
from .configs import DATA_DIR, EXCLUDED_STEMS
from .credentials import resolve
from .feedback_eval import atomic_text, digest, read, write


@dataclass(frozen=True)
class FullExperiment:
    name: str = "qwen38-recovery-v2-full2700"
    artifact_root: str = base.SPEC.artifact_root
    count: int = 2700
    primary_count: int = 2500
    workers: int = 4
    instance_type: str = base.SPEC.instance_type
    format: str = "block"
    report_every: int = 20
    min_credential_seconds: int = 600
    transport_rounds: int = 4
    transport_backoff_seconds: int = 60
    order_seed: int = 1234


experiment_configs = [FullExperiment()]
SPEC = experiment_configs[0]
ARMS = ("history", "canvas", "canvas_last")
TERMINAL = {"completed", "execution_failed", "construction_failed", "format_error"}
TRANSIENT = (
    "APIConnectionError:",
    "APITimeoutError:",
    "RateLimitError:",
    "InternalServerError:",
    "JSONDecodeError:",
)
CSS = "body{font:15px/1.5 system-ui;max-width:1250px;margin:30px auto;padding:16px}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eee;padding:12px}img{max-width:550px;width:100%}details{padding:12px;border:1px solid #ccc;margin:12px 0}input,select{padding:10px;margin:8px}a{color:#245bc0}.bad{background:#fff0ed}"


def prepare(out, spec=SPEC):
    tasks = ds.load_tasks()
    if len(tasks) != spec.count or len({t.task_id for t in tasks}) != spec.count:
        raise ValueError("full benchmark count/uniqueness mismatch")
    primary = [t for t in tasks if t.stem not in EXCLUDED_STEMS]
    if len(primary) != spec.primary_count:
        raise ValueError("primary benchmark count mismatch")

    # Round-robin across stems; within-stem order is independent of outcomes.
    def key(t):
        return hashlib.sha256(f"{spec.order_seed}:{t.task_id}".encode()).hexdigest()

    buckets = {
        stem: sorted([t for t in tasks if t.stem == stem], key=key)
        for stem in sorted({t.stem for t in tasks})
    }
    ordered = [
        bucket[i]
        for i in range(max(map(len, buckets.values())))
        for bucket in buckets.values()
        if i < len(bucket)
    ]
    value = {
        "design": asdict(spec),
        "protocol": asdict(base.SPEC),
        "generation": asdict(base.CFG),
        "tasks": [asdict(t) for t in ordered],
        "issues": EXCLUDED_STEMS,
        "dataset_hashes": {
            name: digest(DATA_DIR / name) for name in sorted({t.file for t in tasks})
        },
    }
    value = json.loads(json.dumps(value))
    path = Path(out) / "inputs.json"
    if path.exists() and read(path) != value:
        raise ValueError("frozen full experiment changed")
    write(path, value)
    extra.write_tls_bundle(out)
    return value


def stamp():
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def arm_order(taskid):
    index = (
        int(hashlib.sha256(f"{SPEC.order_seed}:{taskid}:arms".encode()).hexdigest(), 16)
        % 6
    )
    return list(permutations(ARMS))[index]


def recover_raw(directory):
    """Recover a received response after interruption; never invent a lost reply."""
    directory = Path(directory)
    if (directory / "result.json").exists():
        return
    raws = sorted(directory.glob("raw_*.json"))
    if not raws:
        return
    if len(raws) != 1:
        raise RuntimeError("ambiguous uncommitted raw responses")
    raw = read(raws[0])
    choice = raw["choices"][0]
    write(
        directory / "result.json",
        {
            "content": choice["message"].get("content") or "",
            "error": None,
            "finish_reason": choice.get("finish_reason"),
            "usage": raw.get("usage") or {},
            "wall_time_s": 0,
            "recovered_raw": True,
        },
    )


def full_request(client, messages, schema, fmt, directory, salt):
    directory = Path(directory)
    for attempt in range(SPEC.transport_rounds):
        recover_raw(directory)
        cached = directory / "result.json"
        result = None
        if cached.exists():
            saved = read(directory / "request.json")
            if saved["messages"] != messages:
                raise RuntimeError("resume context differs from cached request")
            result = read(cached)
            if not result.get("error"):
                return result
            if result["error"].startswith("AuthenticationError:"):
                # A prior run may have stopped on an expired key. Retry only
                # the saved rejection; a fresh authentication error stops again.
                archive_request(directory)
                result = None
        if result is None:
            # Refresh credentials at each request; source env file can be renewed
            # while this independent MLS job is running.
            fresh, _ = base._client(base.CFG)
            try:
                result = base.request(fresh, messages, schema, fmt, directory, salt)
            finally:
                fresh.close()
        error = result.get("error") or ""
        if not error or not error.startswith(TRANSIENT):
            return result
        if attempt + 1 == SPEC.transport_rounds:
            return result
        archive_request(directory)
        time.sleep(SPEC.transport_backoff_seconds * (attempt + 1))
    raise AssertionError("unreachable")


def archive_request(directory):
    if list(directory.glob("raw_*.json")):
        raise RuntimeError("refusing to invalidate a received completion")
    archive = directory.parent / "transport_attempts" / directory.name / stamp()
    archive.parent.mkdir(parents=True, exist_ok=True)
    directory.rename(archive)


def read_available(paths):
    # A resumed worker archives nonterminal checkpoints between glob and read.
    # Missing records reappear at the next atomic checkpoint/report.
    rows = []
    for path in paths:
        try:
            rows.append(read(path))
        except FileNotFoundError:
            continue
    return rows


def resume_arm(directory):
    path = Path(directory) / "record.json"
    if not path.exists():
        return
    record = read(path)
    if record["status"] in TERMINAL:
        return
    if record["status"] not in {"running", "request_error", "infrastructure_error"}:
        raise RuntimeError("unknown checkpoint status")
    # Fail closed if a call was in flight and no response was captured. A new
    # generation here would silently reroll an unknown outcome.
    for request in Path(directory).glob("*/request.json"):
        recover_raw(request.parent)
        if not (request.parent / "result.json").exists():
            raise RuntimeError(f"unresolved in-flight request: {request.parent}")
    archive = Path(directory) / "checkpoint_history" / f"{stamp()}.json"
    archive.parent.mkdir(parents=True, exist_ok=True)
    path.rename(archive)


def evaluate_task(value, out, stop):
    task = ds.Task(**value)
    root = Path(out) / "tasks" / task.task_id.replace(":", "_")
    records = [root / a / "record.json" for a in ARMS]
    if all(p.exists() and read(p)["status"] in TERMINAL for p in records):
        return {"task_id": task.task_id, "status": "cached"}
    if stop.is_set():
        return {"task_id": task.task_id, "status": "deferred"}
    try:
        creds = resolve(expected_base_url=base.CFG.base_url)
        if (
            creds.expires_at is not None
            and creds.expires_at - time.time() < SPEC.min_credential_seconds
        ):
            raise RuntimeError(
                "credential near expiry; renew persistent env file then resume"
            )
        first = full_request(
            None,
            base.messages_for(task, SPEC.format),
            base.interactive_schema(),
            SPEC.format,
            root / "first",
            task.task_id + ":recovery:1",
        )
        if first.get("error"):
            raise RuntimeError("first request failed: " + first["error"])
        for arm in arm_order(task.task_id):
            if stop.is_set():
                return {"task_id": task.task_id, "status": "deferred"}
            directory = root / arm
            resume_arm(directory)
            result = base.rollout(
                task, SPEC.format, arm, directory, first, request_fn=full_request
            )
            if result["status"] not in TERMINAL:
                raise RuntimeError(
                    f"{arm}: {result['status']}; inspect saved request errors"
                )
        trace(root)
        (root / "blocked.json").unlink(missing_ok=True)
        return {"task_id": task.task_id, "status": "done"}
    except Exception as exc:
        stop.set()
        error = {
            "task_id": task.task_id,
            "status": "blocked",
            "error": f"{type(exc).__name__}: {exc}",
        }
        write(root / "blocked.json", error)
        return error


def trace(root):
    """Small independent HTML for any task with a terminal non-success."""
    root = Path(root)
    rows = read_available(root / a / "record.json" for a in ARMS)
    if not any(r["status"] in TERMINAL and not r["correct"] for r in rows):
        return
    parts = [
        '<!doctype html><html lang="ru"><meta charset="utf-8"><title>NoReGeo failure</title>',
        f"<style>{CSS}</style>",
        f"<h1>{escape(rows[0]['task_id'])}</h1><p>{escape(rows[0]['question'])}</p>",
        "<p>Эталон: " + escape(str(rows[0]["gold"])) + "</p>",
    ]
    for r in rows:
        parts.append(f"<h2>{r['arm']}: {r['status']}; correct={r['correct']}</h2>")
        for event in r["events"]:
            parts.append(
                "<details><summary>Действие "
                + str(event["call"])
                + "</summary><pre>"
                + escape(event["raw"].get("content") or str(event["raw"].get("error")))
                + "</pre>"
            )
            ex = event.get("execution", {})
            if ex.get("error"):
                parts.append("<pre>" + escape(ex["error"]) + "</pre>")
            if ex.get("image") and base.local_image(ex["image"]).exists():
                parts.append(
                    '<img src="data:image/png;base64,'
                    + base64.b64encode(
                        base.local_image(ex["image"]).read_bytes()
                    ).decode()
                    + '">'
                )
            parts.append("</details>")
        if "final_raw" in r:
            parts.append(
                "<h3>Сырой финальный ответ</h3><pre>"
                + escape(json.dumps(r["final_raw"], ensure_ascii=False, indent=2))
                + "</pre>"
            )
        if r.get("error"):
            parts.append(
                "<p>Ошибка разбора/выполнения:</p><pre>"
                + escape(str(r["error"]))
                + "</pre>"
            )
        parts.append(
            "<p>Финальный ответ: " + escape(str(r.get("final_answer"))) + "</p>"
        )
    parts.append("</html>")
    atomic_text(root / "analysis.html", "\n".join(parts))


def report(out):
    out = Path(out)
    snapshot = read(out / "inputs.json")
    tasks = {extra.task_id(t): t for t in snapshot["tasks"]}
    rows = read_available(out.glob("tasks/*/*/record.json"))
    lookup = {(r["task_id"], r["arm"]): r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError("duplicate task/arm records")
    for r in rows:
        if r["task_id"] not in tasks or r["arm"] not in ARMS:
            raise ValueError("unknown task or arm")
    for r in rows:
        first_path = (
            out / "tasks" / r["task_id"].replace(":", "_") / "first" / "result.json"
        )
        if not first_path.exists() or r["first"] != read(first_path):
            raise ValueError("arm first response differs from canonical first")
    metrics = {}
    for group, ids in [
        (
            "primary",
            [k for k, t in tasks.items() if t["stem"] not in snapshot["issues"]],
        ),
        ("all", list(tasks)),
    ]:
        id_set = set(ids)
        group_rows = [r for r in rows if r["task_id"] in id_set]
        arms = {}
        for arm in ARMS:
            subset = [r for r in group_rows if r["arm"] == arm]
            arms[arm] = last.summarize(subset, len(ids))
            arms[arm]["statuses"] = dict(Counter(r["status"] for r in subset))
            arms[arm]["wrong_answer"] = sum(
                r["status"] == "completed" and not r["correct"] for r in subset
            )
            arms[arm]["execution_failure"] = sum(
                r["status"] in TERMINAL - {"completed"} for r in subset
            )
        paired = {}
        for left, right in [
            ("history", "canvas"),
            ("history", "canvas_last"),
            ("canvas", "canvas_last"),
        ]:
            pairs = []
            for taskid in ids:
                left_row, right_row = (
                    lookup.get((taskid, left)),
                    lookup.get((taskid, right)),
                )
                if (
                    left_row
                    and right_row
                    and left_row["status"] in TERMINAL
                    and right_row["status"] in TERMINAL
                ):
                    pairs.append(
                        tuple(
                            {"checks": {"answer_ok": v["correct"]}}
                            for v in (left_row, right_row)
                        )
                    )
            counts = base.paired_counts(pairs)
            paired[left + " vs " + right] = {
                "n": counts["n"],
                "both": counts["both"],
                "neither": counts["neither"],
                "left_only": counts["history_only"],
                "right_only": counts["canvas_only"],
                "p": counts["mcnemar_exact_p"],
            }
        metrics[group] = {"arms": arms, "paired": paired}
    firsts = read_available(out.glob("tasks/*/first/result.json"))
    continuations = 0
    for r in rows:
        raw = [e["raw"] for e in r["events"][1:]]
        if "final_raw" in r:
            raw.append(r["final_raw"])
        continuations += sum(x.get("usage", {}).get("total_tokens", 0) for x in raw)
    metrics["accounted_tokens"] = (
        sum(r.get("usage", {}).get("total_tokens", 0) for r in firsts) + continuations
    )
    metrics["note"] = (
        "Live record-based usage; responses not yet checkpointed may lag. Transport errors without usage have unknown spend."
    )
    write(out / "comparison.json", metrics)
    parts = [
        '<!doctype html><html lang="ru"><meta charset="utf-8"><title>NoReGeo full benchmark</title>',
        f"<style>{CSS}</style>",
        "<h1>NoReGeo: полный бенчмарк, три варианта</h1>",
        "<p>History — без PNG; canvas — все предыдущие кадры; canvas_last — последний успешный кадр. Текстовая история одинакова по правилам. Все 2700 задач запускаются; основной score исключает 200 задач с известными дефектами датасета. Отказы API и незавершённые измерения показаны отдельно.</p>",
    ]
    for group, title in [
        ("primary", "Основная выборка"),
        ("all", "Все задачи, включая дефекты датасета"),
    ]:
        parts.append(
            f"<h2>{title}</h2><table><tr><th>Вариант</th><th>Верно / план</th><th>Завершено</th><th>Неверный ответ</th><th>Сбой построения</th><th>Входные токены</th><th>Выходные токены</th></tr>"
        )
        for arm, m in metrics[group]["arms"].items():
            terminal = sum(m["statuses"].get(s, 0) for s in TERMINAL)
            parts.append(
                f"<tr><td>{arm}</td><td>{m['correct']}/{m['planned']}</td><td>{terminal}</td><td>{m['wrong_answer']}</td><td>{m['execution_failure']}</td><td>{m['prompt_tokens']}</td><td>{m['completion_tokens']}</td></tr>"
            )
        parts.append("</table>")
    blocked = read_available([out / "blocked.json"])
    if blocked:
        parts.append(
            '<p class="bad">Остановлено с сохранением прогресса: '
            + escape(str(blocked[0]))
            + "</p>"
        )
    parts.append(
        "<p>В строках токенов первый общий ответ учтён в каждом варианте. Известный расход без повторного учёта: "
        + str(metrics["accounted_tokens"])
        + ". Выход включает reasoning.</p>"
    )
    parts.append(
        '<h2>Задачи и ошибки</h2><input id="q" placeholder="Тип, ID или текст задачи"><select id="mode"><option value="fail">Неудачи и расхождения (без дефектов датасета)</option><option value="issue">Дефекты датасета</option><option value="all">Все задачи</option></select><table><thead><tr><th>Задача</th><th>History</th><th>Canvas</th><th>Последний кадр</th><th>Примечание</th></tr></thead><tbody id="cases">'
    )
    for taskid, t in tasks.items():
        rs = [lookup.get((taskid, a)) for a in ARMS]
        issue = snapshot["issues"].get(t["stem"], "")
        failed = any(r and r["status"] in TERMINAL and not r["correct"] for r in rs)
        labels = []
        for r in rs:
            labels.append(
                "…"
                if not r
                else "✓"
                if r["correct"]
                else "ответ ×"
                if r["status"] == "completed"
                else "построение ×"
                if r["status"] in TERMINAL
                else r["status"]
            )
        directory = out / "tasks" / taskid.replace(":", "_")
        target = directory / "analysis.html"
        if failed and not target.exists():
            trace(directory)
        label = (
            f'<a href="{target.relative_to(out)}">{escape(taskid)}</a>'
            if target.exists()
            else escape(taskid)
        )
        parts.append(
            f'<tr data-fail="{int(failed and not issue)}" data-issue="{int(bool(issue))}"><td>{label}<br>{escape(t["row"]["question"])}</td>'
            + "".join("<td>" + escape(x) + "</td>" for x in labels)
            + "<td>"
            + escape(issue)
            + "</td></tr>"
        )
    parts.append(
        """</tbody></table><script>function filter(){let q=document.getElementById('q').value.toLowerCase(),m=document.getElementById('mode').value;document.querySelectorAll('#cases tr').forEach(r=>r.hidden=!(r.textContent.toLowerCase().includes(q)&&(m==='all'||(m==='fail'?r.dataset.fail==='1':r.dataset.issue==='1'))));}document.getElementById('q').oninput=filter;document.getElementById('mode').onchange=filter;filter();</script></html>"""
    )
    atomic_text(out / "analysis.html", "\n".join(parts))
    return metrics


def _run(out):
    out = Path(out)
    snapshot = read(out / "inputs.json")
    expected = json.loads(
        json.dumps(
            {
                "design": asdict(SPEC),
                "protocol": asdict(base.SPEC),
                "generation": asdict(base.CFG),
            }
        )
    )
    if any(snapshot[k] != v for k, v in expected.items()):
        raise ValueError("frozen protocol mismatch")
    if (out / "complete.json").exists():
        report(out)
        return 0
    # Only remove stale blocked marker after a working authenticated startup.
    client, _ = base._client(base.CFG)
    try:
        base.check_model_is_served(client, base.CFG)
    finally:
        client.close()
    (out / "blocked.json").unlink(missing_ok=True)
    stop = threading.Event()
    blocked = []
    done = 0
    started = time.monotonic()
    iterator = iter(snapshot["tasks"])
    with ThreadPoolExecutor(max_workers=SPEC.workers) as pool:
        pending = {}

        def submit():
            try:
                t = next(iterator)
            except StopIteration:
                return False
            pending[pool.submit(evaluate_task, t, out, stop)] = t
            return True

        for _ in range(SPEC.workers):
            submit()
        while pending:
            ready, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in ready:
                pending.pop(future)
                result = future.result()
                done += 1
                if result["status"] == "blocked":
                    blocked.append(result)
                print(json.dumps({"processed_tasks": done, **result}), flush=True)
                if not stop.is_set():
                    submit()
            if done % SPEC.report_every == 0:
                report(out)
    if blocked:
        write(
            out / "blocked.json",
            {"errors": blocked, "wall_time_s": time.monotonic() - started},
        )
        report(out)
        return 2
    metrics = report(out)
    terminal = sum(
        sum(m["statuses"].get(s, 0) for s in TERMINAL)
        for m in metrics["all"]["arms"].values()
    )
    if terminal != SPEC.count * len(ARMS):
        raise RuntimeError("missing/nonterminal measurements")
    write(
        out / "complete.json",
        {
            "status": "all benchmark measurements terminal",
            "wall_time_s": time.monotonic() - started,
        },
    )
    return 0


def run(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare", "run", "report"):
        mode.add_argument("--" + name, action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.prepare:
        prepare(args.out)
        return 0
    if args.report:
        report(args.out)
        return 0
    return run(args.out)


if __name__ == "__main__":
    raise SystemExit(main())
