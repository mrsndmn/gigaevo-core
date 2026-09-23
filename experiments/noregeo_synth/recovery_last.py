"""Compare one-current-frame feedback against frozen history/all-frame baselines."""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from html import escape
import json
from pathlib import Path
import time

from . import recovery_eval as base
from . import recovery_extra as extra
from .dataset import Task
from .feedback_eval import atomic_text, digest, read, write


@dataclass(frozen=True)
class LastFrameExperiment:
    name: str = "qwen38-recovery-v2-last30"
    source: str = str(Path(extra.SPEC.artifact_root) / extra.SPEC.name)
    artifact_root: str = extra.SPEC.artifact_root
    count: int = 30
    workers: int = base.SPEC.workers
    instance_type: str = base.SPEC.instance_type
    format: str = "block"
    arm: str = "canvas_last"


experiment_configs = [LastFrameExperiment()]
SPEC = experiment_configs[0]
ARMS = ("history", "canvas", SPEC.arm)


def prepare(out, spec=SPEC):
    source = Path(spec.source)
    frozen = read(source / "inputs.json")
    tasks = frozen["previous_tasks"] + frozen["tasks"]
    ids = [extra.task_id(t) for t in tasks]
    if len(ids) != spec.count or len(set(ids)) != spec.count:
        raise ValueError("expected exactly30 distinct previous task IDs")
    if frozen["generation"] != asdict(base.CFG) or frozen["protocol"] != json.loads(
        json.dumps(asdict(base.SPEC))
    ):
        raise ValueError("generation/protocol changed from baselines")
    rows = list(frozen["previous_records"])
    hashes = {}
    for t in frozen["tasks"]:
        for arm in base.ARMS:
            p = (
                source
                / "tasks"
                / extra.task_id(t).replace(":", "_")
                / arm
                / "record.json"
            )
            rows.append(read(p))
            hashes[str(p.relative_to(source))] = digest(p)
    lookup = {(r["task_id"], r["arm"]): r for r in rows}
    if len(rows) != 2 * spec.count or len(lookup) != len(rows):
        raise ValueError("missing or duplicate baseline records")
    first = {}
    for t in tasks:
        taskid = extra.task_id(t)
        h, c = [lookup[(taskid, arm)] for arm in base.ARMS]
        for r in (h, c):
            if r["status"] != "completed" or r["format"] != spec.format:
                raise ValueError("baseline not completed in selected format")
            if r["question"] != t["row"]["question"] or r["gold"] != t["row"]["answer"]:
                raise ValueError("baseline task mismatch")
        if h["first"] != c["first"] or h["first"].get("error"):
            raise ValueError("baseline first responses differ or failed")
        first[taskid] = h["first"]
    value = {
        "design": asdict(spec),
        "generation": asdict(base.CFG),
        "protocol": asdict(base.SPEC),
        "tasks": tasks,
        "baselines": rows,
        "first": first,
        "source_inputs_sha256": digest(source / "inputs.json"),
        "source_record_hashes": hashes,
    }
    value = json.loads(json.dumps(value))
    path = Path(out) / "inputs.json"
    if path.exists() and read(path) != value:
        raise ValueError("frozen last-frame comparison changed")
    write(path, value)
    extra.write_tls_bundle(out)
    return value


def summarize(rows, planned):
    normalized = [
        {**r, "events": [{"raw": r["first"]}] + r["events"][1:]} for r in rows
    ]
    result = base.arm_metrics(normalized)
    result["planned"] = planned
    raws = [e["raw"] for r in normalized for e in r["events"]]
    raws += [r["final_raw"] for r in normalized if "final_raw" in r]
    for key in ("prompt_tokens", "completion_tokens"):
        result[key] = sum(raw.get("usage", {}).get(key, 0) for raw in raws)
    result["tokens_per_task"] = round(
        result["total_tokens_including_shared_first"] / planned, 1
    )
    return result


def paired(previous, current):
    lookup = {r["task_id"]: r for r in current}
    pairs = []
    for old in previous:
        new = lookup.get(old["task_id"])
        if new and new["status"] not in {
            "running",
            "request_error",
            "infrastructure_error",
        }:
            pairs.append(
                tuple({"checks": {"answer_ok": r["correct"]}} for r in (old, new))
            )
    counts = base.paired_counts(pairs)
    # Generic labels: base helper calls left/right history/canvas.
    return {
        "n": counts["n"],
        "both": counts["both"],
        "neither": counts["neither"],
        "baseline_only": counts["history_only"],
        "last_only": counts["canvas_only"],
        "mcnemar_exact_p": counts["mcnemar_exact_p"],
    }


def report(out):
    out = Path(out)
    snapshot = read(out / "inputs.json")
    current = [read(p) for p in sorted(out.glob(f"tasks/*/{SPEC.arm}/record.json"))]
    for r in current:
        if r["first"] != snapshot["first"].get(r["task_id"]):
            raise ValueError("new arm first response differs from frozen baseline")
    rows = snapshot["baselines"] + current
    metrics = {
        arm: summarize([r for r in rows if r["arm"] == arm], len(snapshot["tasks"]))
        for arm in ARMS
    }
    matched = {
        arm: paired([r for r in snapshot["baselines"] if r["arm"] == arm], current)
        for arm in base.ARMS
    }
    # First actions were paid for in earlier jobs and are not generated again.
    physical = [read(p) for p in out.glob("tasks/**/result.json")]
    additional = sum(r.get("usage", {}).get("total_tokens", 0) for r in physical)
    images = []
    for p in out.glob("tasks/**/request.json"):
        request = read(p)
        images.append(
            sum(
                item.get("type") == "image_url"
                for message in request["messages"]
                if isinstance(message.get("content"), list)
                for item in message["content"]
            )
        )
    if any(n > 1 for n in images):
        raise ValueError("last-frame request contains multiple images")
    data = {
        "arms": metrics,
        "paired": matched,
        "additional_tokens": additional,
        "new_requests": len(physical),
        "max_images_per_request": max(images, default=0),
    }
    write(out / "comparison.json", data)
    parts = [
        '<!doctype html><html lang="ru"><meta charset="utf-8"><title>NoReGeo: три варианта</title>',
        "<style>body{font:16px/1.5 system-ui;max-width:1150px;margin:30px auto;padding:16px}table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #ccc}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eee;padding:12px}details{border:1px solid #ccc;padding:12px;margin:14px 0}img{max-width:500px;width:100%}</style>",
        "<h1>NoReGeo: history / canvas / canvas_last</h1><p>Одни и те же 30 задач. History получает текст; canvas — все прошлые кадры; canvas_last — только последний успешный кадр. Полная текстовая история сохраняется во всех вариантах, включая ошибки и заменённые действия. После неудачного действия сохраняется последний кадр корректного состояния. Фильтр действует и на запрос финального ответа.</p>",
        "<p>Исторические history/canvas не перезапускались; их результаты зафиксированы. Первый ответ каждой задачи переиспользован дословно. Это последовательные запуски, поэтому изменение серверной модели или нагрузки отдельно не контролируется.</p>",
        "<table><tr><th>Вариант</th><th>Верно / план</th><th>Завершено</th><th>Ошибок действий</th><th>Вход</th><th>Выход</th><th>Всего</th><th>На задачу</th></tr>",
    ]
    for arm, m in metrics.items():
        parts.append(
            f"<tr><td>{arm}</td><td>{m['correct']}/{m['planned']}</td><td>{m['completed']}</td><td>{m['failed_actions']}</td><td>{m['prompt_tokens']}</td><td>{m['completion_tokens']}</td><td>{m['total_tokens_including_shared_first']}</td><td>{m['tokens_per_task']}</td></tr>"
        )
    parts.append(
        f"</table><p>В каждой строке учтён первый общий запрос для сопоставимости; выходные токены включают reasoning. Новые фактически полученные ответы: {len(physical)}, дополнительный расход {additional} токенов. Максимум изображений в одном новом запросе: {data['max_images_per_request']}. Ошибки без usage не позволяют оценить расход.</p>"
    )
    for arm, p in matched.items():
        parts.append(
            f"<p>{arm} vs canvas_last: пар {p['n']}, верен только {arm}: {p['baseline_only']}; только canvas_last: {p['last_only']}; p={p['mcnemar_exact_p']:.4g}.</p>"
        )
    parts.append(
        "<h2>Новые траектории</h2><p>Все кадры сохранены для аудита, но модель получает только последний.</p>"
    )
    for r in current:
        parts.append(
            f"<details><summary>{escape(r['task_id'])}: {r['status']}, correct={r['correct']}</summary><p>{escape(r['question'])}</p>"
        )
        for e in r["events"]:
            parts.append(
                f"<pre>{escape(e['raw'].get('content') or str(e['raw'].get('error')))}</pre>"
            )
            execution = e.get("execution", {})
            if execution.get("error"):
                parts.append("<pre>" + escape(execution["error"]) + "</pre>")
            frame = execution.get("image")
            if frame and base.local_image(frame).exists():
                parts.append(
                    '<img src="data:image/png;base64,'
                    + base64.b64encode(base.local_image(frame).read_bytes()).decode()
                    + '">'
                )
        parts.append(
            f"<p>Ответ: {escape(str(r.get('final_answer')))}; эталон: {escape(str(r['gold']))}</p></details>"
        )
    parts.append("</html>")
    atomic_text(out / "analysis.html", "\n".join(parts))
    return data


def run(out):
    out = Path(out)
    snapshot = read(out / "inputs.json")
    expected = json.loads(
        json.dumps(
            {
                "design": asdict(SPEC),
                "generation": asdict(base.CFG),
                "protocol": asdict(base.SPEC),
            }
        )
    )
    if any(snapshot[k] != v for k, v in expected.items()):
        raise ValueError("worker protocol differs from snapshot")
    if (out / "complete.json").exists():
        report(out)
        return
    client, _ = base._client(base.CFG)
    base.check_model_is_served(client, base.CFG)
    started = time.monotonic()

    def evaluate(t):
        task = Task(**t)
        return base.rollout(
            task,
            SPEC.format,
            SPEC.arm,
            out / "tasks" / task.task_id.replace(":", "_") / SPEC.arm,
            snapshot["first"][task.task_id],
        )

    with ThreadPoolExecutor(max_workers=SPEC.workers) as pool:
        for _ in pool.map(evaluate, snapshot["tasks"]):
            report(out)
    write(
        out / "complete.json",
        {
            "status": "all30 new-arm measurements attempted",
            "wall_time_s": time.monotonic() - started,
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
