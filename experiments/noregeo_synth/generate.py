"""Generate the synthetic "reasoning + incremental plot" dataset from NoReGeo.

For each benchmark task the model returns a JSON solution: N reasoning steps,
each with a matplotlib fragment that adds that step's construction to a shared
canvas. We execute the fragments in order (one PNG per step), then keep only
samples that survive every check in `verify.py`.

Usage (single-sample smoke run):

    python -m experiments.noregeo_synth.generate --task outer_circle_center:6200

    python -m experiments.noregeo_synth.generate \\
        --config qwen38-27b --n 20 --stem triangle_area --out run/noregeo_synth

Everything about *what* is generated lives in `configs.py`.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import sys
import threading
import time
from typing import Any

from . import dataset as ds
from .canvas import build_preamble
from .configs import (
    CONFIGS_BY_NAME,
    DEFAULT_CONFIG,
    DEFAULT_WORKERS,
    OUT_DIR,
    SMOKE_TASK,
    GenConfig,
)
from .credentials import resolve
from .execute import run_steps
from .prompts import build_messages, repair_messages, response_schema
from .verify import answer_matches, compare_frames, contact_sheet, literal_leak

HEARTBEAT_SECONDS = 300.0


def _client(cfg: GenConfig):
    from openai import OpenAI

    creds = resolve(expected_base_url=cfg.base_url)
    days = creds.expires_in_days
    if days is not None and days < 3:
        print(f"[warn] inference token expires in {days:.1f} days", file=sys.stderr)
    # max_retries=1: the SDK's default of 2 composes with our own retry loop and
    # with the repair rounds, so one sample could pin a worker for hours.
    return OpenAI(
        base_url=cfg.base_url,
        api_key=creds.api_key,
        timeout=cfg.request_timeout,
        max_retries=1,
    ), creds


def _seed_for(cfg: GenConfig, salt: str, attempt: int) -> int | None:
    """Deterministic per-task seed.

    Derived from the task id rather than being one global constant, so that a
    resumed run does not replay the exact seed sequence that already failed —
    the whole premise of re-rolling is that the same seed reproduces the same
    degenerate sample.
    """
    if cfg.seed is None:
        return None
    digest = int(hashlib.sha1(salt.encode()).hexdigest()[:8], 16)
    return (cfg.seed + digest + 7919 * attempt) % (2**31 - 1)


def _call(
    client,
    cfg: GenConfig,
    messages: list[dict],
    schema: dict,
    salt: str = "",
):
    """One completion, retried while it comes back unusable.

    Unusable means empty content (all the budget went into hidden thinking) or
    unparseable JSON. Both normally end with `finish_reason="length"`, but for
    two different reasons that need two different retries: the sampler
    degenerating into whitespace (re-roll the seed — a bigger budget just burns
    more), or a solution that genuinely did not fit (raise the budget — a new
    seed will not make it shorter). So the first retry re-rolls, and later ones
    also escalate `max_tokens` up to `max_tokens_ceiling`.

    Tokens spent on the failed attempts are returned too: they are most of the
    cost of exactly the samples the retry exists for, and dropping them would
    make the records under-report spend.
    """
    last: Exception | None = None
    spent: dict[str, Any] = {}
    budget = cfg.max_tokens
    for attempt in range(cfg.call_retries + 1):
        kwargs: dict[str, Any] = dict(
            model=cfg.model,
            messages=messages,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=budget,
            response_format={"type": "json_schema", "json_schema": schema},
        )
        seed = _seed_for(cfg, salt, attempt)
        if seed is not None:
            kwargs["seed"] = seed
        if cfg.thinking == "off":
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        elif cfg.thinking:
            kwargs["extra_body"] = {"reasoning_effort": cfg.thinking}

        resp = client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        msg = choice.message
        thinking = getattr(msg, "reasoning", None) or getattr(
            msg, "reasoning_content", None
        )
        usage = resp.usage.model_dump() if resp.usage else {}
        try:
            if not msg.content:
                raise RuntimeError(
                    f"empty completion (finish_reason={choice.finish_reason}, "
                    f"thinking chars={len(thinking or '')})"
                )
            return json.loads(msg.content), thinking, _add_usage(spent, usage)
        except (RuntimeError, json.JSONDecodeError) as exc:
            last = exc
            _add_usage(spent, usage)
            print(
                f"[warn] unusable completion (attempt {attempt + 1}/"
                f"{cfg.call_retries + 1}, {len(msg.content or '')} chars, "
                f"finish={choice.finish_reason}, budget={budget}): {exc}",
                file=sys.stderr,
            )
            if choice.finish_reason == "length" and attempt >= 1:
                budget = min(budget * 2, cfg.max_tokens_ceiling)
    raise RuntimeError(
        f"no usable completion after {cfg.call_retries + 1} attempts "
        f"(last budget {budget}): {last}"
    )


def _add_usage(total: dict, extra: dict) -> dict:
    """Sum token counters across repair rounds, nested dicts included."""
    for key, value in (extra or {}).items():
        if isinstance(value, dict):
            total[key] = _add_usage(
                total.get(key) if isinstance(total.get(key), dict) else {}, value
            )
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            base = total.get(key)
            total[key] = (base if isinstance(base, (int, float)) else 0) + value
        elif key not in total:
            total[key] = value
    return total


def golden_point(row: dict[str, Any]) -> tuple[float, float] | None:
    """The benchmark's answer as an (x, y) pair, for point-schema tasks."""
    if row.get("schema_type") != "point":
        return None
    gold = row.get("answer")
    xy = gold.get("point") if isinstance(gold, dict) else gold
    try:
        return (float(xy[0]), float(xy[1]))
    except (TypeError, ValueError, IndexError):
        return None


def run_checks(
    row: dict[str, Any], solution: dict, exec_res, cfg: GenConfig
) -> dict[str, Any]:
    """Every acceptance signal for one candidate solution, plus what failed.

    A signal that could not be evaluated (because nothing ran, or because the
    task's answer schema does not support it) is None, never True: an unchecked
    box must not read as a passed one.
    """
    problems: list[str] = []

    if not exec_res.ok:
        failed = exec_res.first_failure
        detail = exec_res.error or (failed.error if failed else "unknown failure")
        if failed is not None and failed.index <= len(solution["steps"]):
            code = solution["steps"][failed.index - 1]["code"]
            detail = f"Step {failed.index} code:\n```python\n{code}\n```\n\n{detail}"
        problems.append(f"The code did not run:\n{detail}")

    step_frames = [s.frame for s in exec_res.steps if s.ok and s.image]
    comparisons = (
        compare_frames(step_frames, baseline=exec_res.base_frame) if step_frames else []
    )
    deltas = [c["delta"] for c in comparisons]
    losses = [c["ink_loss"] for c in comparisons]

    incremental_ok: bool | None = None
    if exec_res.ok:
        dead = [
            i
            for i, d in enumerate(deltas, 1)
            if d is None or not (d >= cfg.min_pixel_delta)
        ]
        erased = [
            i
            for i, ink in enumerate(losses, 1)
            if ink is not None and ink > cfg.max_ink_loss
        ]
        incremental_ok = not dead and not erased
        if dead:
            problems.append(
                f"Step(s) {dead} drew nothing new (or changed the figure size) — "
                "every step must add something visible to the shared canvas."
            )
        if erased:
            problems.append(
                f"Step(s) {erased} erased what earlier steps had drawn. Steps may "
                "only add; do not hide, cover or delete existing content."
            )

    ans_ok, ans_detail = answer_matches(row, solution.get("final_answer"), cfg)
    if exec_res.ok and not ans_ok:
        problems.append(f"`final_answer` is wrong: {ans_detail}.")

    code_ok: bool | None = None
    code_detail = "not checked (the code did not run)"
    if exec_res.ok:
        if exec_res.state_answer is None:
            code_ok = False
            code_detail = (
                'the last step never set state["answer"] '
                f"(state keys: {exec_res.state_keys})"
            )
        else:
            code_ok, code_detail = answer_matches(row, exec_res.state_answer, cfg)
        if not code_ok:
            problems.append(
                f"The answer your CODE computed is wrong: {code_detail}. "
                "The prose and the drawing must agree with each other."
            )

    # Known weakness: in 10 of the 900 point rows a *given* point lies within
    # draw_atol of the golden one, so marking the givens satisfies this check
    # without the answer ever being drawn. The state["answer"] check still has
    # to pass for those, so nothing is accepted on the drawing alone.
    drawn_ok: bool | None = None
    draw_dist = exec_res.answer_point_distance
    if exec_res.ok and row.get("schema_type") == "point":
        drawn_ok = draw_dist is not None and draw_dist <= cfg.draw_atol
        if not drawn_ok:
            where = (
                "no point is marked at all"
                if draw_dist is None
                else f"the nearest marked point is {draw_dist:.3g} away"
            )
            problems.append(
                f"The answer is not drawn where it belongs: {where}. "
                "Mark the answer point at its true coordinates."
            )

    leaks = literal_leak(row, solution["steps"]) if cfg.reject_pasted_answer else []
    if leaks and cfg.reject_pasted_answer:
        problems.append(
            "The answer is pasted into the code instead of computed: "
            + "; ".join(leaks)
            + ". Derive it from the constructions of the earlier steps."
        )

    return {
        "deltas": deltas,
        "ink_losses": losses,
        "incremental_ok": incremental_ok,
        "answer_ok": ans_ok if exec_res.ok else None,
        "answer_detail": ans_detail,
        "code_answer_ok": code_ok,
        "code_answer_detail": code_detail,
        "drawn_ok": drawn_ok,
        "draw_distance": draw_dist,
        "literal_leaks": leaks,
        "problems": problems,
    }


def generate_one(
    task: ds.Task,
    cfg: GenConfig,
    out_root: Path,
    client=None,
    keep_reference: bool = True,
) -> dict[str, Any]:
    """Generate, execute and verify one sample. Returns the dataset record."""
    row = task.row
    if client is None:
        client, _ = _client(cfg)

    sample_dir = out_root / cfg.name / f"{task.stem}_{task.qid}"
    if sample_dir.exists():
        shutil.rmtree(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)

    preamble = build_preamble(row, cfg)
    (sample_dir / "preamble.py").write_text(preamble)
    schema = response_schema(row, cfg)
    messages = build_messages(row, cfg, preamble)
    (sample_dir / "prompt.json").write_text(
        json.dumps(messages, ensure_ascii=False, indent=2)
    )

    target = golden_point(row)

    t0 = time.time()
    solution, thinking, usage = _call(client, cfg, messages, schema, salt=task.task_id)
    thinkings = [thinking]
    exec_res = run_steps(preamble, solution["steps"], sample_dir, cfg, target)
    checks = run_checks(row, solution, exec_res, cfg)

    rounds = 0
    while checks["problems"] and rounds < cfg.max_repair_rounds:
        rounds += 1
        solution, thinking, usage_round = _call(
            client,
            cfg,
            repair_messages(messages, solution, "\n\n".join(checks["problems"])),
            schema,
            salt=f"{task.task_id}#repair{rounds}",
        )
        thinkings.append(thinking)
        usage = _add_usage(usage, usage_round)
        exec_res = run_steps(preamble, solution["steps"], sample_dir, cfg, target)
        checks = run_checks(row, solution, exec_res, cfg)

    images = [s.image for s in exec_res.steps if s.ok and s.image]
    deltas = checks["deltas"]
    losses = checks["ink_losses"]

    steps_out = []
    for i, step in enumerate(solution["steps"], 1):
        res = next((s for s in exec_res.steps if s.index == i), None)
        steps_out.append(
            {
                "index": i,
                "title": step["title"],
                "reasoning": step["reasoning"],
                "code": step["code"],
                "image": (
                    str(Path(res.image).relative_to(sample_dir))
                    if res and res.image
                    else None
                ),
                "exec_ok": bool(res and res.ok),
                "error": res.error if res else "not executed",
                "pixel_delta": deltas[i - 1] if i - 1 < len(deltas) else None,
                "ink_loss": losses[i - 1] if i - 1 < len(losses) else None,
            }
        )

    reference = None
    if keep_reference and row.get("im_path"):
        try:
            src = ds.fetch_image(row["im_path"])
            dst = sample_dir / "reference_full.png"
            shutil.copyfile(src, dst)
            reference = dst.name
        except Exception as exc:  # network hiccup must not kill a run
            print(f"[warn] reference image unavailable: {exc}", file=sys.stderr)

    if images:
        contact_sheet(
            images,
            [
                f"step {i}: {s['title']}"
                for i, s in enumerate(steps_out, 1)
                if s["image"]
            ],
            sample_dir / "contact_sheet.png",
            suptitle=f"{task.task_id} — {row.get('question', '')[:110]}",
        )

    record = {
        "task_id": task.task_id,
        "qid": task.qid,
        "source_file": task.file,
        "category": row.get("category"),
        "difficulty": row.get("difficulty"),
        "schema_type": row.get("schema_type"),
        "question": row.get("question"),
        "choices": row.get("choices"),
        "points": row.get("points"),
        "golden_answer": row.get("answer"),
        "config": cfg.name,
        "model": cfg.model,
        "reveal_answer": cfg.reveal_answer,
        "plan": solution.get("plan"),
        "steps": steps_out,
        "final_answer": solution.get("final_answer"),
        "answer_ok": checks["answer_ok"],
        "answer_detail": checks["answer_detail"],
        "code_answer": exec_res.state_answer,
        "code_answer_ok": checks["code_answer_ok"],
        "code_answer_detail": checks["code_answer_detail"],
        "drawn_ok": checks["drawn_ok"],
        "draw_distance": checks["draw_distance"],
        "literal_leaks": checks["literal_leaks"],
        "exec_ok": exec_res.ok,
        "exec_error": exec_res.error,
        "incremental_ok": checks["incremental_ok"],
        "accepted": not checks["problems"],
        "problems": checks["problems"],
        "repair_rounds": rounds,
        "thinking": thinkings[-1],
        "thinking_rounds": thinkings,
        "usage": usage,
        "wall_time_s": round(time.time() - t0, 1),
        "preamble": "preamble.py",
        "base_image": "canvas_base.png" if exec_res.base_frame else None,
        "reference_image": reference,
        "sample_dir": str(sample_dir),
    }
    (sample_dir / "record.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2)
    )
    return record


def read_records(records_path: Path) -> list[dict[str, Any]]:
    rows = []
    if records_path.exists():
        for line in records_path.read_text().splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


class RecordStore:
    """`records.jsonl` as a keyed store: one row per (task_id, config).

    Keyed rather than append-only because the sample directory is rewritten
    from scratch on every generation — an append log leaves earlier rows
    pointing at images that belong to a later run, i.e. a dataset whose
    reasoning and figures silently disagree.

    Rows are held in memory and the file is rewritten at most every
    `flush_interval` seconds. Rewriting per sample is O(n^2): measured on this
    NFS mount, a 2700-sample run would spend ~8 minutes inside the write lock
    and ~26 GB of write traffic on it. Each sample also keeps its own
    `record.json`, so a crash between flushes loses bookkeeping, not data.
    """

    def __init__(self, path: Path, flush_interval: float = 10.0) -> None:
        self.path = path
        self.flush_interval = flush_interval
        self.lock = threading.Lock()
        self.rows: dict[tuple[str, str], dict[str, Any]] = {
            (r.get("task_id"), r.get("config")): r for r in read_records(path)
        }
        self._last_flush = time.time()

    def add(self, record: dict[str, Any]) -> None:
        with self.lock:
            self.rows[(record["task_id"], record["config"])] = record
            if time.time() - self._last_flush >= self.flush_interval:
                self._flush()

    def close(self) -> None:
        with self.lock:
            self._flush()

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Unique tmp name: two runs writing the same --out must not clobber
        # each other's half-written file.
        tmp = self.path.with_suffix(f".jsonl.tmp{os.getpid()}")
        tmp.write_text(
            "".join(
                json.dumps(r, ensure_ascii=False) + "\n" for r in self.rows.values()
            )
        )
        tmp.replace(self.path)
        self._last_flush = time.time()


def upsert_record(records_path: Path, record: dict[str, Any]) -> None:
    """Single-record convenience wrapper around `RecordStore`."""
    store = RecordStore(records_path, flush_interval=0.0)
    store.add(record)
    store.close()


def failure_record(
    task: ds.Task, cfg: GenConfig, out_root: Path, exc: Exception
) -> dict:
    """A row for a sample that never produced a solution.

    Without this, a failure after `generate_one` has already wiped the sample
    directory leaves the *previous* run's `accepted: true` row in place,
    pointing at PNGs that no longer exist — and `--resume` then skips the task
    forever.
    """
    row = task.row
    return {
        "task_id": task.task_id,
        "qid": task.qid,
        "source_file": task.file,
        "category": row.get("category"),
        "difficulty": row.get("difficulty"),
        "schema_type": row.get("schema_type"),
        "question": row.get("question"),
        "golden_answer": row.get("answer"),
        "config": cfg.name,
        "model": cfg.model,
        "steps": [],
        "accepted": False,
        "exec_ok": False,
        "problems": [f"generation failed: {type(exc).__name__}: {exc}"],
        "error": f"{type(exc).__name__}: {exc}",
        "sample_dir": str(out_root / cfg.name / f"{task.stem}_{task.qid}"),
    }


def _artifacts_present(row: dict[str, Any]) -> bool:
    """Does this row's sample directory still hold what the row claims?"""
    sample_dir = Path(row.get("sample_dir") or "")
    if not (sample_dir / "record.json").exists():
        return False
    return all(
        (sample_dir / step["image"]).exists()
        for step in row.get("steps", [])
        if step.get("image")
    )


def accepted_task_ids(records_path: Path) -> set[str]:
    """Tasks already generated, accepted, and still present on disk.

    The on-disk check is what makes `--resume` safe: a row whose images were
    wiped by a later interrupted attempt has to be regenerated, not skipped.
    """
    return {
        row["task_id"]
        for row in read_records(records_path)
        if row.get("accepted") and _artifacts_present(row)
    }


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _stratified(tasks: list[ds.Task], key, per_group: int, seed: int) -> list[ds.Task]:
    """`per_group` tasks from every group, drawn with a per-group seed."""
    groups: dict[Any, list[ds.Task]] = {}
    for task in tasks:
        groups.setdefault(key(task), []).append(task)
    out: list[ds.Task] = []
    for i, name in enumerate(sorted(groups, key=str)):
        pool = groups[name]
        if per_group > len(pool):
            print(
                f"[warn] {name}: asked for {per_group}, only {len(pool)} exist",
                file=sys.stderr,
            )
        out += random.Random(seed + i).sample(pool, min(per_group, len(pool)))
    return out


def _select_tasks(tasks: list[ds.Task], args) -> list[ds.Task]:
    selected: list[ds.Task] = []
    if args.per_stem:
        selected += _stratified(tasks, lambda t: t.stem, args.per_stem, args.seed)
    if args.per_category:
        selected += _stratified(
            tasks, lambda t: t.row.get("category"), args.per_category, args.seed
        )
    if args.n:
        pool = [t for t in tasks if not args.stem or t.stem == args.stem]
        if not pool:
            stems = sorted({t.stem for t in tasks})
            raise SystemExit(
                f"--stem {args.stem!r} matches no task file. Available: "
                + ", ".join(stems)
            )
        if args.n > len(pool):
            print(
                f"[warn] --n {args.n} exceeds the pool of {len(pool)}; "
                f"generating {len(pool)}",
                file=sys.stderr,
            )
        selected += random.Random(args.seed).sample(pool, min(args.n, len(pool)))
    explicit = args.task or ([] if selected else [SMOKE_TASK])
    for spec in explicit:
        selected.append(ds.select(tasks, spec))

    seen: set[str] = set()
    unique = []
    for task in selected:
        if task.task_id not in seen:
            seen.add(task.task_id)
            unique.append(task)
    if len(unique) != len(selected):
        print(
            f"[warn] dropped {len(selected) - len(unique)} duplicate selector(s)",
            file=sys.stderr,
        )
    return unique


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=DEFAULT_CONFIG, choices=list(CONFIGS_BY_NAME))
    ap.add_argument(
        "--task",
        action="append",
        help='sample selector "<stem>:<qid>" (repeatable). '
        f"Defaults to the smoke sample {SMOKE_TASK}.",
    )
    ap.add_argument("--stem", help="restrict random sampling to one task file")
    ap.add_argument("--n", type=int, default=0, help="random samples to draw")
    ap.add_argument(
        "--per-stem",
        type=int,
        default=0,
        help="draw N tasks from each of the 27 task files (each holds 100)",
    )
    ap.add_argument(
        "--per-category",
        type=int,
        default=0,
        help="draw N tasks from each of the 9 `category` values",
    )
    ap.add_argument("--seed", type=int, default=0, help="sampling seed")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--model", help="override the config's model id")
    ap.add_argument("--no-reveal-answer", action="store_true")
    ap.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="concurrent samples in flight (the endpoint batches them)",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="skip tasks already accepted in records.jsonl",
    )
    args = ap.parse_args(argv)

    cfg = CONFIGS_BY_NAME[args.config]
    if args.model:
        # The model is part of the arm's identity: without renaming, two models
        # would share one output directory and one records.jsonl.
        cfg = dataclasses.replace(
            cfg, model=args.model, name=f"{cfg.name}-{_slug(args.model)}"
        )
    if args.no_reveal_answer:
        cfg = dataclasses.replace(cfg, reveal_answer=False)

    tasks = ds.load_tasks()
    selected = _select_tasks(tasks, args)
    records_path = args.out / cfg.name / "records.jsonl"

    if args.resume:
        done = accepted_task_ids(records_path)
        before = len(selected)
        selected = [t for t in selected if t.task_id not in done]
        print(
            f"resume: {before - len(selected)} already accepted, {len(selected)} left"
        )

    client, creds = _client(cfg)
    workers = max(1, min(args.workers, len(selected) or 1))
    print(
        f"model={cfg.model} endpoint={cfg.base_url} creds={creds.source} "
        f"samples={len(selected)} workers={workers}"
    )

    store = RecordStore(records_path)
    lock = threading.Lock()
    started = time.time()
    counters = {"done": 0, "accepted": 0, "failed": 0}
    in_flight: dict[str, float] = {}

    def report(task: ds.Task, rec: dict, note: str) -> None:
        with lock:
            store.add(rec)
            in_flight.pop(task.task_id, None)
            counters["done"] += 1
            counters["accepted"] += bool(rec.get("accepted"))
            counters["failed"] += not rec.get("accepted")
            done, total = counters["done"], len(selected)
            # An ETA from fewer completions than there are workers is noise:
            # the first finisher makes it look like the batch is serial.
            eta = (
                f" eta={(total - done) * (time.time() - started) / done / 60:.0f}m"
                if done >= workers
                else ""
            )
            print(
                f"[{done}/{total} acc={counters['accepted']}{eta}] "
                f"{task.task_id:34s} {note}",
                flush=True,
            )

    def work(task: ds.Task) -> None:
        with lock:
            in_flight[task.task_id] = time.time()
        try:
            rec = generate_one(task, cfg, args.out, client=client)
        except Exception as exc:  # one bad completion must not kill the batch
            report(
                task,
                failure_record(task, cfg, args.out, exc),
                f"ERROR {type(exc).__name__}: {exc}",
            )
            return
        flags = " ".join(
            f"{k}={'ok' if rec[k] else 'n/a' if rec[k] is None else 'FAIL'}"
            for k in (
                "exec_ok",
                "incremental_ok",
                "answer_ok",
                "code_answer_ok",
                "drawn_ok",
            )
        )
        report(
            task,
            rec,
            f"steps={len(rec['steps'])} repairs={rec['repair_rounds']} {flags}",
        )

    def heartbeat() -> None:
        """Something to look at in the log while a slow sample is in flight."""
        while not finished.wait(HEARTBEAT_SECONDS):
            with lock:
                stuck = sorted(
                    ((time.time() - t0, tid) for tid, t0 in in_flight.items()),
                    reverse=True,
                )
            if stuck:
                oldest = ", ".join(f"{tid} ({age / 60:.1f}m)" for age, tid in stuck[:3])
                print(
                    f"[heartbeat {counters['done']}/{len(selected)}] "
                    f"{len(stuck)} in flight, oldest: {oldest}",
                    flush=True,
                )

    finished = threading.Event()
    beat = threading.Thread(target=heartbeat, daemon=True)
    beat.start()
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(work, selected))
    finally:
        finished.set()
        store.close()

    elapsed = (time.time() - started) / 60
    print(
        f"accepted {counters['accepted']}/{len(selected)}"
        + (f", {counters['failed']} not accepted" if counters["failed"] else "")
        + f" in {elapsed:.1f} min -> {records_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
