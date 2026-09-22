"""Resumable paired evaluation on a frozen historical NoReGeo cohort.

--prepare snapshots inputs, --run performs inference, --report rebuilds HTML.
All design decisions are defined by experiment_configs below.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
from html import escape
import json
import math
from pathlib import Path
import tempfile
import threading

from . import dataset as ds
from .configs import EVAL_CONFIGS_BY_NAME, EXCLUDED_STEMS, REPO_ROOT
from .eval_blind import evaluate_one
from .feedback import _BASE, rollout
from .feedback import experiment_configs as feedback_configs
from .feedback_report import CSS, build_report
from .generate import _client, batch_stopper, check_model_is_served


@dataclass(frozen=True)
class EvalExperiment:
    name: str = "qwen38-historical75-v1"
    historical_file: str = "run/noregeo_eval/qwen38-27b/records.jsonl"
    historical_sha256: str = (
        "daeaba7c8f7290ae8c1f743a9273b965a5aaa1d23f7baa4ce00071cd7edb2a5d"
    )
    baseline_config: str = "qwen38-27b"
    task_count: int = 75
    workers: int = 2
    instance_type: str = "a100.1gpu"  # live allocation discovery, 2026-09-23
    artifact_root: str = "/mnt/virtual_ai0001053-00054_SR004-nfs2/d.tarasov/gigaevo-artifacts/results/noregeo_feedback"


experiment_configs = [EvalExperiment()]
SPEC = experiment_configs[0]
TERMINAL = {"completed", "execution_failed"}


def read(path):
    return json.loads(Path(path).read_text())


def atomic_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=path.name + ".", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(content)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def design(spec=SPEC):
    return {
        "experiment": asdict(spec),
        "feedback": [asdict(c) for c in feedback_configs],
        "fresh_text": asdict(EVAL_CONFIGS_BY_NAME[spec.baseline_config]),
    }


def cohort(historical, pool, spec=SPEC):
    """Select by membership in the old text arm, never by correctness."""
    previous = [r for r in historical if r.get("arm") == "text"]
    ids = [r["task_id"] for r in previous]
    if len(ids) != spec.task_count or len(set(ids)) != len(ids):
        raise ValueError("historical cohort size/uniqueness mismatch")
    lookup = {t.task_id: t for t in pool}
    tasks = []
    for record in sorted(previous, key=lambda r: r["task_id"]):
        task = lookup[record["task_id"]]
        if task.stem in EXCLUDED_STEMS:
            raise ValueError(f"excluded task in frozen cohort: {task.task_id}")
        if (
            record["question"] != task.row["question"]
            or record["golden_answer"] != task.row["answer"]
        ):
            raise ValueError(f"benchmark changed: {task.task_id}")
        tasks.append(task)
    return tasks


def prepare(out, source_root=REPO_ROOT, spec=SPEC):
    out = Path(out)
    original = Path(source_root) / spec.historical_file
    if digest(original) != spec.historical_sha256:
        raise ValueError("historical results checksum changed; define a new experiment")
    historical = [
        json.loads(line) for line in original.read_text().splitlines() if line.strip()
    ]
    tasks = cohort(historical, ds.load_tasks(), spec)
    snapshot = {
        "design": design(spec),
        "tasks": [asdict(t) for t in tasks],
        "historical": historical,
    }
    target = out / "inputs.json"
    if target.exists():
        if read(target) != snapshot:
            raise ValueError("existing input snapshot differs")
    else:
        write(target, snapshot)
    return snapshot


def task_directory(out, task_id):
    return Path(out) / "tasks" / task_id.replace(":", "_")


def arm_directory(out, task_id, arm):
    return task_directory(out, task_id) / "rollouts" / "sample" / arm


def stored(out, task_id, arm):
    path = (
        task_directory(out, task_id) / "text.json"
        if arm == "text"
        else arm_directory(out, task_id, arm) / "record.json"
    )
    try:
        return read(path)
    except FileNotFoundError:
        return None


def recover_first_step(out, task_id):
    shared_file = task_directory(out, task_id) / "first_step.json"
    candidates = [read(shared_file)] if shared_file.exists() else []
    for arm in ("history", "canvas"):
        record = stored(out, task_id, arm)
        if record and record.get("steps"):
            candidates.append(record["steps"][0])
    if any(candidate != candidates[0] for candidate in candidates):
        raise ValueError(f"conflicting first actions: {task_id}")
    if candidates:
        write(shared_file, candidates[0])
        return candidates[0]
    return None


def finished(record, arm):
    return bool(record) and (
        not record.get("error") if arm == "text" else record.get("status") in TERMINAL
    )


def paired_counts(records):
    counts = {"both": 0, "history_only": 0, "canvas_only": 0, "neither": 0}
    for h, c in records:
        a = bool((h.get("checks") or {}).get("answer_ok"))
        b = bool((c.get("checks") or {}).get("answer_ok"))
        key = (
            "both"
            if a and b
            else "history_only"
            if a
            else "canvas_only"
            if b
            else "neither"
        )
        counts[key] += 1
    discordant = counts["history_only"] + counts["canvas_only"]
    k = min(counts["history_only"], counts["canvas_only"])
    counts["mcnemar_exact_p"] = (
        min(
            1.0, 2 * sum(math.comb(discordant, i) for i in range(k + 1)) / 2**discordant
        )
        if discordant
        else 1.0
    )
    counts["n"] = sum(
        counts[k] for k in ("both", "history_only", "canvas_only", "neither")
    )
    return counts


def comparison(out):
    """Fixed denominators and explicitly paired historical subsets; offline only."""
    out = Path(out)
    snapshot = read(out / "inputs.json")
    ids = [ds.Task(**t).task_id for t in snapshot["tasks"]]
    old = {(r["task_id"], r["arm"]): r for r in snapshot["historical"]}
    rows, pairs, protocol_errors = [], [], []
    for arm in ("text", "history", "canvas"):
        records = [stored(out, tid, arm) for tid in ids]
        done = [r for r in records if finished(r, arm)]
        correct = sum(
            bool(
                r.get("ok")
                if arm == "text"
                else (r.get("checks") or {}).get("answer_ok")
            )
            for r in done
        )
        rows.append(
            {
                "arm": arm,
                "planned": len(ids),
                "finished": len(done),
                "correct": correct,
                "errors": sum(
                    bool(r and not finished(r, arm) and r.get("error")) for r in records
                ),
                "accepted": None
                if arm == "text"
                else sum(bool(r.get("accepted")) for r in done),
            }
        )
    for tid in ids:
        h, c = stored(out, tid, "history"), stored(out, tid, "canvas")
        if finished(h, "history") and finished(c, "canvas"):
            if (
                not h.get("steps")
                or not c.get("steps")
                or h["steps"][0] != c["steps"][0]
            ):
                protocol_errors.append(tid)
            else:
                pairs.append((h, c))
    matched = []
    for old_arm in ("text", "both", "figure", "swap", "swap-control"):
        # Swapped configurations have different gold answers: descriptive only,
        # never silently included in the same-target comparison.
        if old_arm in ("swap", "swap-control"):
            continue
        eligible = [
            tid
            for tid in ids
            if (tid, old_arm) in old
            and not any(
                old[tid, old_arm].get(k)
                for k in (
                    "error",
                    "coords_injected",
                    "figure_underspecified",
                    "leaked_coordinates",
                )
            )
        ]
        for arm in ("text", "history", "canvas"):
            common = [tid for tid in eligible if finished(stored(out, tid, arm), arm)]
            new_correct = sum(
                bool(
                    stored(out, tid, arm).get("ok")
                    if arm == "text"
                    else (stored(out, tid, arm).get("checks") or {}).get("answer_ok")
                )
                for tid in common
            )
            matched.append(
                {
                    "old_arm": old_arm,
                    "new_arm": arm,
                    "n": len(common),
                    "old_correct": sum(
                        bool(old[tid, old_arm].get("ok")) for tid in common
                    ),
                    "new_correct": new_correct,
                }
            )
    data = {
        "rows": rows,
        "paired": paired_counts(pairs),
        "matched": matched,
        "protocol_errors": protocol_errors,
    }
    write(out / "comparison.json", data)
    html = [
        '<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        f"<title>NoReGeo — сравнение экспериментов</title><style>{CSS}table{{width:100%;border-collapse:collapse;background:white}}td,th{{padding:10px;border-bottom:1px solid #ddd;text-align:left}}.scroll{{overflow:auto}}</style><main>",
        '<div class="eyebrow">NoReGeo / парный eval</div><h1>Сравнение со старым экспериментом</h1>',
        "<p>Основной эффект: canvas против history на одинаковых задачах и с общим первым действием. Свежий text повторяет прежний протокол. Исторические text/both/figure — ориентиры с другим протоколом и бюджетом, не чистая абляция.</p>",
        "<p>Правильность — строгий допуск старого eval. Ошибки исполнения считаются неудачными решениями; ошибки запросов показаны отдельно и не исчезают из планового объёма. До завершения это счётчик успехов, не итоговая точность.</p>",
        '<div class="scroll"><table><tr><th>Новый режим</th><th>Завершено / план</th><th>Верных ответов</th><th>Принятых трейсов</th><th>Ошибок запросов/прогона</th></tr>',
    ]
    for row in rows:
        html.append(
            f"<tr><td>{row['arm']}</td><td>{row['finished']} / {row['planned']}</td><td>{row['correct']}</td><td>{row['accepted'] if row['accepted'] is not None else '—'}</td><td>{row['errors']}</td></tr>"
        )
    if protocol_errors:
        html.append(
            "<p>Нарушен общий первый шаг; пары исключены: "
            + escape(", ".join(protocol_errors))
            + "</p>"
        )
    p = data["paired"]
    html += [
        "</table></div><h2>Парное сравнение пошаговых режимов</h2>",
        f"<p>Завершённых пар: {p['n']}. Оба верны: {p['both']}; только history: {p['history_only']}; только canvas: {p['canvas_only']}; оба неверны: {p['neither']}.</p>",
        f"<p>Точный двусторонний тест Мак-Немара: p = {p['mcnemar_exact_p']:.4g}. Промежуточное значение не используйте для остановки по значимости.</p>",
        '<h2>Старые и новые результаты на общих task ID</h2><div class="scroll"><table><tr><th>Старый режим</th><th>Новый режим</th><th>Общих задач</th><th>Верных раньше</th><th>Верных сейчас</th></tr>',
    ]
    for r in matched:
        html.append(
            f"<tr><td>{r['old_arm']}</td><td>{r['new_arm']}</td><td>{r['n']}</td><td>{r['old_correct']}</td><td>{r['new_correct']}</td></tr>"
        )
    html += [
        "</table></div><p>figure использует изображение вместо координат в тексте; both — исходное изображение вместе с текстом. swap исключён: он меняет целевую конфигурацию. Совпадение task ID не делает протоколы эквивалентными.</p>",
        "<h2>Разбор задач</h2><ul>",
    ]
    for tid in ids:
        path = task_directory(out, tid) / "analysis.html"
        html.append(
            f'<li><a href="{escape(str(path.relative_to(out)))}">{escape(tid)}</a></li>'
            if path.exists()
            else f"<li>{escape(tid)} — ожидает прогона</li>"
        )
    html += ["</ul></main></html>"]
    atomic_text(out / "comparison.html", "".join(html))
    return data


def evaluate_task(task, out, stop, spec=SPEC):
    directory = task_directory(out, task.task_id)
    directory.mkdir(parents=True, exist_ok=True)
    baseline = EVAL_CONFIGS_BY_NAME[spec.baseline_config]
    # Client is per task/thread; limits bound total simultaneous endpoint calls.
    client, _ = _client(_BASE)
    if not finished(stored(out, task.task_id, "text"), "text") and not stop.is_set():
        try:
            result = evaluate_one(task, "text", baseline, directory, client)
        except Exception as exc:
            result = {
                "task_id": task.task_id,
                "arm": "text",
                "error": f"{type(exc).__name__}: {exc}",
                "ok": False,
            }
            write(directory / "text.json", result)
            if batch_stopper(exc):
                stop.set()
                raise
        write(directory / "text.json", result)
    shared_file = directory / "first_step.json"
    shared = recover_first_step(out, task.task_id)
    for arm in feedback_configs:
        if stop.is_set():
            break
        existing = stored(out, task.task_id, arm.name)
        if finished(existing, arm.name):
            if shared is None and existing.get("steps"):
                shared = existing["steps"][0]
                write(shared_file, shared)
            continue
        destination = arm_directory(out, task.task_id, arm.name)
        if destination.exists():
            # Preserve old attempt and reuse its first action on resume.
            if shared is None and existing and existing.get("steps"):
                shared = existing["steps"][0]
                write(shared_file, shared)
            archive = (
                directory
                / "attempts"
                / f"{arm.name}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%f')}"
            )
            archive.parent.mkdir(exist_ok=True)
            destination.rename(archive)
        try:
            result = rollout(task, arm, destination, client, first_step=shared)
        except Exception as exc:
            result = (
                read(destination / "record.json")
                if (destination / "record.json").exists()
                else None
            )
            if batch_stopper(exc):
                stop.set()
                raise
        if shared is None and result and result.get("steps"):
            shared = result["steps"][0]
            write(shared_file, shared)
    if any((directory / "rollouts").glob("*/*/record.json")):
        build_report(directory / "rollouts", directory / "analysis.html")
    print(f"Finished task attempt: {task.task_id}", flush=True)


def run(out, spec=SPEC):
    out = Path(out)
    snapshot = read(out / "inputs.json")
    if snapshot["design"] != design(spec):
        raise ValueError("run design differs from frozen snapshot")
    tasks = [ds.Task(**t) for t in snapshot["tasks"]]
    client, _ = _client(_BASE)
    check_model_is_served(client, _BASE)
    stop = threading.Event()
    report_lock = threading.Lock()

    def work(task):
        try:
            if not stop.is_set():
                evaluate_task(task, out, stop, spec)
        finally:
            with report_lock:
                comparison(out)

    try:
        with ThreadPoolExecutor(max_workers=spec.workers) as executor:
            list(executor.map(work, tasks))
    finally:
        comparison(out)
    complete = all(
        finished(stored(out, t.task_id, arm), arm)
        for t in tasks
        for arm in ("text", "history", "canvas")
    )
    complete = complete and not comparison(out)["protocol_errors"]
    if complete:
        write(
            out / "complete.json",
            {"tasks": len(tasks), "inputs_sha256": digest(out / "inputs.json")},
        )
    return 0 if complete else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--report", action="store_true")
    args = parser.parse_args(argv)
    if args.prepare:
        prepare(args.out)
        comparison(args.out)
        return 0
    if args.report:
        print(json.dumps(comparison(args.out), ensure_ascii=False))
        return 0
    return run(args.out)


if __name__ == "__main__":
    raise SystemExit(main())
