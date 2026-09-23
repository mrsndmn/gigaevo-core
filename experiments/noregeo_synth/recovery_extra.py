"""A separately authorized second recovery cohort; no old gate overrides."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from html import escape
import json
from pathlib import Path
import time

from . import recovery_eval as base
from .feedback_eval import atomic_text, digest, read, write


@dataclass(frozen=True)
class AdditionalExperiment:
    name: str = "qwen38-recovery-v2-extra15"
    artifact_root: str = base.SPEC.artifact_root
    source: str = str(Path(base.SPEC.artifact_root) / base.SPEC.name)
    count: int = 15
    workers: int = base.SPEC.workers
    instance_type: str = base.SPEC.instance_type
    format: str = "block"


experiment_configs = [AdditionalExperiment()]
SPEC = experiment_configs[0]


def task_id(t):
    return t["stem"] + ":" + str(t["row"]["qid"])


def select_tasks(tasks, previous, count):
    """Outcome blind: prioritize previously uncovered stems, one task per stem."""
    previous_ids = {task_id(t) for t in previous}
    counts = Counter(t["stem"] for t in previous)
    remaining = sorted(
        (t for t in tasks if task_id(t) not in previous_ids),
        key=lambda t: (t["stem"], t["row"]["qid"]),
    )
    selected = []
    while len(selected) < count:
        if not remaining:
            raise ValueError("not enough distinct additional tasks")
        chosen = min(
            remaining, key=lambda t: (counts[t["stem"]], t["stem"], t["row"]["qid"])
        )
        selected.append(chosen)
        counts[chosen["stem"]] += 1
        remaining.remove(chosen)
    return selected


def prepare(out, spec=SPEC):
    source = Path(spec.source)
    frozen = read(source / "inputs.json")
    previous = frozen["pilot"]
    if len(previous) != 15:
        raise ValueError("expected completed original 15-task pilot")
    if frozen["generation"] != asdict(base.CFG) or frozen["design"] != json.loads(
        json.dumps(asdict(base.SPEC))
    ):
        raise ValueError("original protocol differs from current code")
    selected = select_tasks(frozen["tasks"], previous, spec.count)
    old_records = []
    hashes = {}
    for task in previous:
        for arm in base.ARMS:
            path = (
                source / "tasks" / task_id(task).replace(":", "_") / arm / "record.json"
            )
            row = read(path)
            if row["status"] != "completed" or row["format"] != spec.format:
                raise ValueError("original pilot not complete in expected format")
            old_records.append(row)
            hashes[str(path.relative_to(source))] = digest(path)
    snapshot = {
        "design": asdict(spec),
        "protocol": asdict(base.SPEC),
        "generation": asdict(base.CFG),
        "tasks": selected,
        "pilot": selected,
        "cases": [],
        "injection": [],
        "previous_tasks": previous,
        "previous_records": old_records,
        "source_record_hashes": hashes,
        "source_inputs_sha256": digest(source / "inputs.json"),
    }
    snapshot = json.loads(json.dumps(snapshot))
    path = Path(out) / "inputs.json"
    if path.exists() and read(path) != snapshot:
        raise ValueError("frozen additional cohort changed")
    write(path, snapshot)
    return snapshot


def group_metrics(rows, planned):
    arms = {
        arm: base.arm_metrics([r for r in rows if r["arm"] == arm]) for arm in base.ARMS
    }
    for arm in base.ARMS:
        arms[arm]["planned"] = planned
        arms[arm]["tokens_per_planned_task"] = round(
            arms[arm]["total_tokens_including_shared_first"] / planned, 1
        )
        arms[arm]["prompt_tokens"] = 0
        arms[arm]["completion_tokens"] = 0
        for row in rows:
            if row["arm"] != arm:
                continue
            raws = [e["raw"] for e in row["events"]]
            if "final_raw" in row:
                raws.append(row["final_raw"])
            for raw in raws:
                for field in ("prompt_tokens", "completion_tokens"):
                    arms[arm][field] += raw.get("usage", {}).get(field, 0)
    # The initial response is used twice but paid for only once.
    firsts = {}
    for row in rows:
        if row["task_id"] in firsts and row["first"] != firsts[row["task_id"]]:
            raise ValueError("paired first responses differ")
        firsts[row["task_id"]] = row["first"]
    shared = sum(r.get("usage", {}).get("total_tokens", 0) for r in firsts.values())
    by_id = {}
    for row in rows:
        by_id.setdefault(row["task_id"], {})[row["arm"]] = row
    pairs = []
    for pair in by_id.values():
        if all(a in pair for a in base.ARMS) and all(
            pair[a]["status"]
            not in {"running", "request_error", "infrastructure_error"}
            for a in base.ARMS
        ):
            pairs.append(
                tuple({"checks": {"answer_ok": pair[a]["correct"]}} for a in base.ARMS)
            )
    # Subtract only duplicated shared responses, including incomplete pairs correctly.
    duplicated = sum(
        pair["history"]["first"].get("usage", {}).get("total_tokens", 0)
        for pair in by_id.values()
        if all(a in pair for a in base.ARMS)
    )
    return {
        "arms": arms,
        "shared_first_tokens": shared,
        "actual_tokens": sum(
            a["total_tokens_including_shared_first"] for a in arms.values()
        )
        - duplicated,
        "paired": base.paired_counts(pairs),
    }


def report(out):
    out = Path(out)
    snapshot = read(out / "inputs.json")
    base.report(out)
    current = [read(p) for p in sorted(out.glob("tasks/*/*/record.json"))]
    previous = snapshot["previous_records"]
    data = {
        "first15": group_metrics(previous, len(snapshot["previous_tasks"])),
        "new15": group_metrics(current, len(snapshot["tasks"])),
        "all30": group_metrics(
            previous + current, len(snapshot["previous_tasks"]) + len(snapshot["tasks"])
        ),
    }
    write(out / "comparison.json", data)
    parts = [
        '<!doctype html><html lang="ru"><meta charset="utf-8"><title>NoReGeo: 30 задач</title>',
        "<style>body{font:16px/1.5 system-ui;max-width:1200px;margin:30px auto;padding:15px}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ccc;text-align:left}</style>",
        "<h1>NoReGeo: дополнительные 15 задач</h1><p>Qwen3.8-27B, прежний протокол и отдельный Python-блок. Наборы не пересекаются; выбор задач не учитывает ответы. Результаты первого пилота зафиксированы в снимке входов.</p>",
    ]
    for key, title in [
        ("first15", "Первый пилот"),
        ("new15", "Новые задачи"),
        ("all30", "Объединённый набор"),
    ]:
        group = data[key]
        parts.append(
            f"<h2>{title}</h2><table><tr><th>Вариант</th><th>Верно / план</th><th>Завершено</th><th>Ошибок действий</th><th>Вход</th><th>Выход</th><th>Всего токенов</th><th>На задачу</th></tr>"
        )
        for arm, m in group["arms"].items():
            parts.append(
                f"<tr><td>{arm}</td><td>{m['correct']}/{m['planned']}</td><td>{m['completed']}</td><td>{m['failed_actions']}</td><td>{m['prompt_tokens']}</td><td>{m['completion_tokens']}</td><td>{m['total_tokens_including_shared_first']}</td><td>{m['tokens_per_planned_task']}</td></tr>"
            )
        pair = group["paired"]
        parts.append(
            f"</table><p>Расход без двойного учёта первого запроса: {group['actual_tokens']} токенов. Пар: {pair['n']}; только history: {pair['history_only']}; только canvas: {pair['canvas_only']}; p={pair['mcnemar_exact_p']:.4g}.</p>"
        )
    parts.append(
        "<p>Выходные токены включают reasoning. В каждой строке учтён общий первый запрос. Токены изображений входят во входной usage API. Неудачные попытки учитываются; расходы неизвестны, если API не вернул usage. Форматная диагностика первого запуска не входит в эти таблицы.</p>"
    )
    parts.append("<h2>Новые задачи</h2><ul>")
    for t in snapshot["tasks"]:
        parts.append(f"<li>{escape(task_id(t))}: {escape(t['row']['question'])}</li>")
    parts.append("</ul></html>")
    atomic_text(out / "comparison.html", "\n".join(parts))
    return data


def run(out):
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
        raise ValueError("worker design differs from frozen inputs")
    if (out / "complete.json").exists():
        report(out)
        return
    client, _ = base._client(base.CFG)
    base.check_model_is_served(client, base.CFG)
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=SPEC.workers) as pool:
        for _ in pool.map(
            lambda t: base.evaluate_pair(t, SPEC.format, out), snapshot["tasks"]
        ):
            report(out)
    write(
        out / "complete.json",
        {
            "status": "all additional measurements attempted",
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
