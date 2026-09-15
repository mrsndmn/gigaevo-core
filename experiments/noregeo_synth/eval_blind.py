"""Coordinate-blind evaluation of VLMs on NoReGeo, plus a falsification test.

The question this answers is not "can the model do geometry" — with the
coordinates in the text, that is arithmetic. It is "can the model get the
configuration out of a picture", which is the capability the synthesis dataset
cannot teach on its own, because the generator never saw an image.

Five arms (see `configs.ARMS`):

    text          the benchmark's text question, coordinates and all, no image
    figure        the benchmark's `image_question` + its dots figure: the model
                  must read the points off the picture
    both          text question + figure: does the picture add anything
    swap          text question with the ORIGINAL coordinates + a figure of the
                  same configuration scaled by `SWAP_SCALE`, and the prompt says
                  the figure is authoritative. An answer for the scaled
                  configuration means the model read the picture; an answer for
                  the original means it read the text.
    swap-control  the same, at scale 1.0, so that "follows the text" can be told
                  apart from "cannot parse our renderer"

Usage:

    python -m experiments.noregeo_synth.eval_blind --config qwen38-27b --per-stem 2
    python -m experiments.noregeo_synth.eval_blind --arm figure --arm swap --per-stem 20
    python -m experiments.noregeo_synth.eval_blind --summary            # re-print

Everything about *what* is evaluated lives in `configs.py`.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import dataclasses
from pathlib import Path
import re
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any

from . import dataset as ds
from .configs import (
    ARM_BOTH,
    ARM_FIGURE,
    ARM_SWAP,
    ARM_SWAP_CONTROL,
    ARMS,
    DEFAULT_EVAL_CONFIG,
    DEFAULT_EVAL_WORKERS,
    EVAL_CONFIGS_BY_NAME,
    EVAL_IMAGE_FIELD,
    EVAL_OUT_DIR,
    EXCLUDED_STEMS,
    FIGURE_ANSWER_IN_TITLE,
    FIGURE_UNDERSPECIFIED,
    QUESTION_REPAIRS,
    SWAP_RULE,
    SWAP_SCALE,
    EvalConfig,
)
from .figures import iter_points, render_dots, scale_points, union_view_box

# Reused rather than reimplemented: the retry policy (`_call`), the "is this
# failure about the environment?" taxonomy and the record store all exist
# because a 2700-task run found their edge cases the hard way.
from .generate import (
    RecordStore,
    _call,
    _client,
    _slug,
    _stratified,
    batch_stopper,
    check_model_is_served,
    free_gb,
    read_records,
)
from .prompts import _answer_format_hint, _answer_schema
from .verify import answer_matches

FIGURE_ARMS = (ARM_FIGURE, ARM_BOTH, ARM_SWAP, ARM_SWAP_CONTROL)
RENDERED_ARMS = (ARM_SWAP, ARM_SWAP_CONTROL)

SYSTEM_TEXT = (
    "You are a highly capable AI assistant with expertise in geometry. You "
    "solve problems precisely and answer in the requested JSON schema."
)
SYSTEM_FIGURE = (
    "You are a highly capable AI assistant with expertise in visual geometry. "
    "You read geometric figures accurately — the labelled points, their "
    "coordinates on the axes, and what the title says about the task — and "
    "solve problems from what the picture shows."
)

READ_OFF = (
    "The coordinates are NOT given in words. Read the position of every "
    "labelled point off the grid in the image, state what you read in "
    "`reasoning`, then compute the answer from those coordinates."
)
FIGURE_IS_AUTHORITATIVE = (
    "IMPORTANT: the coordinates written in the problem text are STALE — the "
    "configuration has since moved. The IMAGE is authoritative. Read the "
    "positions of the labelled points off the image, ignore the coordinates in "
    "the text, and answer for the configuration that is actually drawn."
)

NUM = r"-?\d+(?:\.\d+)?"
PAIR_RE = re.compile(rf"[\[(]?\s*({NUM})\s*,\s*({NUM})\s*[\])]?")


# ---------------------------------------------------------------------------
# Building one request
# ---------------------------------------------------------------------------


def coordinate_leak(text: str, points: dict[str, Any]) -> list[str]:
    """Configuration coordinates that appear verbatim in `text`.

    The whole premise of the figure arm is that the prompt does not contain the
    numbers the model is supposed to read off the picture, so this is checked
    rather than assumed — a leak turns a vision task back into arithmetic and
    would show up as a suspiciously good score.
    """
    wanted = {(round(x, 6), round(y, 6)) for _, (x, y) in iter_points(points)}
    hits = []
    for m in PAIR_RE.finditer(text or ""):
        pair = (round(float(m.group(1)), 6), round(float(m.group(2)), 6))
        if pair in wanted:
            hits.append(m.group(0).strip())
    return hits


def repair_image_question(task: ds.Task) -> tuple[str, dict[str, Any]]:
    """`image_question`, with an unrendered template filled from the text.

    Two stems ship a vision question the benchmark never finished rendering
    (`{axis}`, `None`), which leaves the question turning on a datum that is in
    neither the text nor the picture. The datum is lifted out of the *text*
    question; whether that puts coordinates back into the prompt is recorded,
    because for `symmetry` it does and those rows must not sit in the headline
    number.
    """
    question = task.row.get("image_question") or ""
    meta: dict[str, Any] = {
        "repaired": None,
        "coords_injected": False,
        "figure_underspecified": task.stem in FIGURE_UNDERSPECIFIED,
    }
    spec = QUESTION_REPAIRS.get(task.stem)
    if not spec or str(spec["placeholder"]) not in question:
        return question, meta
    match = re.search(str(spec["pattern"]), task.row.get("question") or "")
    if not match:
        meta["repaired"] = "FAILED"
        return question, meta
    meta["repaired"] = match.group(1)
    meta["coords_injected"] = bool(spec["injects_coordinates"])
    return question.replace(str(spec["placeholder"]), match.group(1)), meta


def swapped_answer(task: ds.Task, scale: float) -> Any:
    """The golden answer for the configuration scaled by `scale`.

    `SWAP_RULE` says how each stem's answer transforms; stems missing from it
    are scale-invariant and never reach this function.
    """
    rule = SWAP_RULE[task.stem]
    gold = task.row.get("answer")
    if rule == "area":
        return float(gold) * scale**2
    if rule == "length":
        return float(gold) * scale
    if rule == "point":
        xy = gold.get("point") if isinstance(gold, dict) else gold
        scaled = [float(xy[0]) * scale, float(xy[1]) * scale]
        return {"point": scaled} if isinstance(gold, dict) else scaled
    raise ValueError(f"unknown swap rule {rule!r} for {task.stem}")


def arm_applies(task: ds.Task, arm: str) -> str | None:
    """None if the arm can run on this task, else why it cannot."""
    if task.stem in EXCLUDED_STEMS:
        return EXCLUDED_STEMS[task.stem]
    if arm in (ARM_FIGURE, ARM_BOTH) and task.stem in FIGURE_ANSWER_IN_TITLE:
        return "the dots figure states the answer in its title"
    if arm in RENDERED_ARMS and task.stem not in SWAP_RULE:
        return "scale-invariant answer (nothing to discriminate)"
    if arm in RENDERED_ARMS and not ds.named_points(task.row):
        return "no structured points to redraw"
    if arm in (ARM_FIGURE, ARM_BOTH) and not task.row.get(EVAL_IMAGE_FIELD):
        return f"no {EVAL_IMAGE_FIELD}"
    return None


def _data_url(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def build_request(
    task: ds.Task, arm: str, cfg: EvalConfig, out_dir: Path
) -> dict[str, Any]:
    """Messages, response schema, golden answer and bookkeeping for one job."""
    row = task.row
    points = ds.named_points(row)
    meta: dict[str, Any] = {
        "arm": arm,
        "repaired": None,
        "coords_injected": False,
        "figure_underspecified": False,
        "image": None,
        "scale": None,
        "leaked_coordinates": [],
    }

    if arm == ARM_FIGURE:
        question, rmeta = repair_image_question(task)
        meta.update(rmeta)
        # `image_choices` goes with `image_question`; the two are parallel
        # fields, and the scoring answer (`answer`) is identical to
        # `image_answer` on every one of the 2700 rows.
        ask_row = {
            **row,
            "question": question,
            "choices": row.get("image_choices") or row.get("choices"),
        }
        instructions = READ_OFF
    else:
        question = row.get("question") or ""
        ask_row = row
        instructions = FIGURE_IS_AUTHORITATIVE if arm in RENDERED_ARMS else ""

    golden = row.get("answer")
    image_path: Path | None = None

    if arm in (ARM_FIGURE, ARM_BOTH):
        image_path = ds.fetch_image(str(row[EVAL_IMAGE_FIELD]))
    elif arm in RENDERED_ARMS:
        scale = SWAP_SCALE if arm == ARM_SWAP else 1.0
        meta["scale"] = scale
        # One view box for both arms, wide enough for BOTH scales. Deriving it
        # from the drawn configuration alone would make the scaled figure
        # pixel-identical to the control apart from the tick labels; deriving
        # it from the unscaled one alone pushed shrunken configurations off the
        # canvas (see `union_view_box`).
        view = union_view_box([points, scale_points(points, SWAP_SCALE)])
        drawn = scale_points(points, scale) if scale != 1.0 else points
        image_path = out_dir / "figures" / f"{task.stem}_{task.qid}_{arm}.png"
        render_dots(drawn, f"Task {task.qid}: {task.stem}", image_path, view=view)
        if scale != 1.0:
            golden = swapped_answer(task, scale)

    if arm == ARM_FIGURE:
        meta["leaked_coordinates"] = coordinate_leak(question, points)

    hint = _answer_format_hint(ask_row)
    text_parts = [question]
    if ask_row.get("choices"):
        text_parts.append(f"Options: {ask_row['choices']}")
    if instructions:
        text_parts.append(instructions)
    text_parts.append(hint)
    text_parts.append(
        "Put your working in `reasoning` (3 sentences at most) and the answer "
        "in `final_answer`."
    )
    prompt = "\n\n".join(p for p in text_parts if p)

    content: Any = prompt
    if image_path is not None:
        meta["image"] = str(image_path)
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": _data_url(image_path)}},
        ]

    messages = [
        {
            "role": "system",
            "content": SYSTEM_FIGURE if arm in FIGURE_ARMS else SYSTEM_TEXT,
        },
        {"role": "user", "content": content},
    ]
    schema = {
        "name": "noregeo_answer",
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "final_answer": _answer_schema(ask_row),
            },
            "required": ["reasoning", "final_answer"],
            "additionalProperties": False,
        },
    }
    return {
        "messages": messages,
        "schema": schema,
        "golden": golden,
        "question": question,
        "meta": meta,
        "ask_row": ask_row,
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _loose(cfg: EvalConfig) -> SimpleNamespace:
    return SimpleNamespace(
        num_rtol=cfg.loose_num_rtol,
        num_atol=cfg.loose_num_atol,
        point_atol=cfg.loose_point_atol,
    )


def score_answer(row: dict[str, Any], golden: Any, final: Any, cfg: EvalConfig) -> dict:
    """Strict and loose match of one answer against one golden value."""
    scoring_row = {**row, "answer": golden}
    strict, detail = answer_matches(scoring_row, final, cfg)
    loose, _ = answer_matches(scoring_row, final, _loose(cfg))
    return {"ok": strict, "close": loose, "detail": detail}


def evaluate_one(
    task: ds.Task, arm: str, cfg: EvalConfig, out_dir: Path, client
) -> dict[str, Any]:
    req = build_request(task, arm, cfg, out_dir)
    started = time.time()
    payload, thinking, usage = _call(
        client,
        cfg,
        req["messages"],
        req["schema"],
        salt=f"{task.task_id}:{arm}",
    )
    final = payload.get("final_answer")
    scored = score_answer(req["ask_row"], req["golden"], final, cfg)

    record: dict[str, Any] = {
        # RecordStore keys on (task_id, config); the arm is part of the identity
        # of a row, so it goes into `config` as well as its own field.
        "task_id": task.task_id,
        "config": f"{cfg.name}:{arm}",
        "arm": arm,
        "eval_config": cfg.name,
        "model": cfg.model,
        "qid": task.qid,
        "stem": task.stem,
        "category": task.row.get("category"),
        "difficulty": task.row.get("difficulty"),
        "schema_type": task.row.get("schema_type"),
        "question": req["question"],
        "golden_answer": req["golden"],
        "final_answer": final,
        "reasoning": payload.get("reasoning"),
        "ok": scored["ok"],
        "close": scored["close"],
        "detail": scored["detail"],
        "usage": usage,
        "wall_time_s": round(time.time() - started, 1),
        "error": None,
        **req["meta"],
    }
    if thinking:
        record["thinking_chars"] = len(thinking)

    # The falsification arm scores the same answer twice: against the figure and
    # against the stale text. Which one it follows is the measurement.
    if arm in RENDERED_ARMS:
        text_gold = task.row.get("answer")
        text_side = score_answer(req["ask_row"], text_gold, final, cfg)
        record["follows_figure"] = scored["ok"]
        record["follows_text"] = text_side["ok"]
        # Read-off error puts a correct reading just outside the strict
        # tolerance, which under strict-only scoring reads as "followed
        # neither" — the one bucket that means nothing.
        record["follows_figure_close"] = scored["close"]
        record["follows_text_close"] = text_side["close"]
        record["text_answer"] = text_gold
        # ...but a loose tolerance only discriminates when the two goldens are
        # further apart than the tolerance itself. They are not on the small
        # answers (incircle radii run 1-3, and `loose_num_atol` is 0.5), where
        # the unscaled answer passes as the scaled one and a row would be
        # counted under BOTH loose columns. Scoring the text golden as if it
        # were the model's answer is exactly that test.
        record["loose_separable"] = not score_answer(
            req["ask_row"], req["golden"], text_gold, cfg
        )["close"]
    return record


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _pct(num: int, den: int) -> str:
    return f"{num / den * 100:5.1f}%" if den else "    — "


def summarise(rows: list[dict[str, Any]], by_stem: bool = False) -> str:
    """Arms in rows, metrics in columns; headline excludes the flagged rows."""
    out: list[str] = []
    arms = [a for a in ARMS if any(r.get("arm") == a for r in rows)]

    header = (
        f"{'arm':14s} {'n':>5s} {'exact':>7s} {'close':>7s} "
        f"{'errors':>7s} {'excluded':>9s}"
    )
    out.append(header)
    out.append("-" * len(header))
    for arm in arms:
        got = [r for r in rows if r.get("arm") == arm]
        # Rows whose prompt had to be repaired with coordinates, or whose figure
        # cannot carry the question, are reported apart from the headline rather
        # than dropped silently.
        flagged = {
            id(r)
            for r in got
            if r.get("coords_injected")
            or r.get("figure_underspecified")
            or r.get("leaked_coordinates")
        }
        clean = [r for r in got if id(r) not in flagged]
        scored = [r for r in clean if not r.get("error")]
        errors = len(clean) - len(scored)
        out.append(
            f"{arm:14s} {len(scored):5d} "
            f"{_pct(sum(bool(r['ok']) for r in scored), len(scored)):>7s} "
            f"{_pct(sum(bool(r['close']) for r in scored), len(scored)):>7s} "
            f"{errors:7d} {len(flagged):9d}"
        )

    swap = [r for r in rows if r.get("arm") in RENDERED_ARMS and not r.get("error")]
    if swap:
        out.append("")
        out.append("falsification (which source the answer follows)")
        head = (
            f"{'arm':14s} {'n':>5s} {'figure':>8s} {'text':>8s} {'neither':>8s}"
            f"{'figure~':>9s} {'text~':>8s}{'ambig':>9s}"
        )
        out.append(head)
        out.append("-" * len(head))
        for arm in RENDERED_ARMS:
            got = [r for r in swap if r["arm"] == arm]
            if not got:
                continue
            fig = sum(bool(r.get("follows_figure")) for r in got)
            txt = sum(bool(r.get("follows_text")) for r in got)
            neither = sum(
                not r.get("follows_figure") and not r.get("follows_text") for r in got
            )
            # The loose columns are computed only where the two goldens are
            # further apart than the loose tolerance; elsewhere they would
            # count the same row twice.
            sep = [r for r in got if r.get("loose_separable")]
            fig_c = sum(bool(r.get("follows_figure_close")) for r in sep)
            txt_c = sum(bool(r.get("follows_text_close")) for r in sep)
            out.append(
                f"{arm:14s} {len(got):5d} {_pct(fig, len(got)):>8s} "
                f"{_pct(txt, len(got)):>8s} {_pct(neither, len(got)):>8s}"
                f"{_pct(fig_c, len(sep)):>9s} {_pct(txt_c, len(sep)):>8s}"
                f"{len(got) - len(sep):9d}"
            )
        out.append(
            "  `~` = loose tolerance, over the rows where the two goldens are "
            "further apart than that tolerance; the rest are counted under "
            "`ambig`. On swap-control the figure and the text agree by "
            "construction, so every row is ambiguous and the arm exists only to "
            "show the model can answer at all with our renderer in the prompt."
        )

    if by_stem:
        out.append("")
        head = f"{'stem':26s}" + "".join(f"{a:>14s}" for a in arms)
        out.append(head)
        out.append("-" * len(head))
        for stem in sorted({r.get("stem", "") for r in rows}):
            cells = []
            for arm in arms:
                got = [
                    r
                    for r in rows
                    if r.get("stem") == stem
                    and r.get("arm") == arm
                    and not r.get("error")
                ]
                cells.append(
                    f"{_pct(sum(bool(r['ok']) for r in got), len(got))}({len(got)})".rjust(
                        14
                    )
                )
            out.append(f"{stem:26s}" + "".join(cells))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _select_tasks(tasks: list[ds.Task], args) -> list[ds.Task]:
    import random

    # `--stem` narrows every selector below it, including the fallback. Without
    # that, `--stem X` on its own quietly evaluated one task from each of the 27
    # stems — 27 tasks' worth of paid requests on the tasks not asked for.
    pool = [t for t in tasks if not args.stem or t.stem == args.stem]
    if args.stem and not pool:
        stems = ", ".join(sorted({t.stem for t in tasks}))
        raise SystemExit(
            f"--stem {args.stem!r} matches no task file. Available: {stems}"
        )

    selected: list[ds.Task] = []
    if args.per_stem:
        selected += _stratified(pool, lambda t: t.stem, args.per_stem, args.seed)
    if args.n:
        if args.n > len(pool):
            print(
                f"[warn] --n {args.n} exceeds the pool of {len(pool)}; "
                f"evaluating {len(pool)}",
                file=sys.stderr,
            )
        selected += random.Random(args.seed).sample(pool, min(args.n, len(pool)))
    for spec in args.task or []:
        try:
            selected.append(ds.select(tasks, spec))
        except KeyError as exc:
            raise SystemExit(f"--task {spec!r}: {exc}") from exc
    if not selected:
        selected = _stratified(pool, lambda t: t.stem, 1, args.seed)

    seen: set[str] = set()
    unique = []
    for task in selected:
        if task.task_id not in seen:
            seen.add(task.task_id)
            unique.append(task)
    return unique


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config", default=DEFAULT_EVAL_CONFIG, choices=list(EVAL_CONFIGS_BY_NAME)
    )
    ap.add_argument("--model", help="override the config's model id")
    ap.add_argument(
        "--arm",
        action="append",
        choices=list(ARMS),
        help="arm to run (repeatable); default: all",
    )
    ap.add_argument("--task", action="append", help='selector "<stem>:<qid>"')
    ap.add_argument("--stem", help="restrict random sampling to one task file")
    ap.add_argument("--n", type=int, default=0, help="random tasks to draw")
    ap.add_argument(
        "--per-stem", type=int, default=0, help="draw N tasks from each task file"
    )
    ap.add_argument("--seed", type=int, default=0, help="sampling seed")
    ap.add_argument("--out", type=Path, default=EVAL_OUT_DIR)
    ap.add_argument("--workers", type=int, default=DEFAULT_EVAL_WORKERS)
    ap.add_argument("--resume", action="store_true", help="skip rows already scored")
    ap.add_argument("--summary", action="store_true", help="print the summary and exit")
    ap.add_argument("--by-stem", action="store_true", help="add a per-stem breakdown")
    args = ap.parse_args(argv)

    cfg = EVAL_CONFIGS_BY_NAME[args.config]
    if args.model:
        cfg = dataclasses.replace(
            cfg, model=args.model, name=f"{cfg.name}-{_slug(args.model)}"
        )
    out_dir = args.out / cfg.name
    records_path = out_dir / "records.jsonl"

    if args.summary:
        rows = read_records(records_path)
        if not rows:
            raise SystemExit(f"no records at {records_path}")
        print(summarise(rows, by_stem=args.by_stem))
        return 0

    arms = args.arm or list(ARMS)
    tasks = ds.load_tasks()
    selected = _select_tasks(tasks, args)

    jobs: list[tuple[ds.Task, str]] = []
    skipped: dict[str, int] = {}
    for task in selected:
        for arm in arms:
            why = arm_applies(task, arm)
            if why:
                skipped[f"{arm}: {why}"] = skipped.get(f"{arm}: {why}", 0) + 1
                continue
            jobs.append((task, arm))
    for reason, count in sorted(skipped.items()):
        print(f"[skip] {count} task(s) — {reason}")

    if args.resume:
        done = {
            (r.get("task_id"), r.get("arm"))
            for r in read_records(records_path)
            if not r.get("error")
        }
        before = len(jobs)
        jobs = [j for j in jobs if (j[0].task_id, j[1]) not in done]
        print(f"resume: {before - len(jobs)} already scored, {len(jobs)} left")

    if not jobs:
        print("nothing to do")
        return 0

    client, creds = _client(cfg)
    check_model_is_served(client, cfg)
    workers = max(1, min(args.workers, len(jobs)))
    print(
        f"model={cfg.model} endpoint={cfg.base_url} creds={creds.source} "
        f"tasks={len(selected)} jobs={len(jobs)} arms={','.join(arms)} "
        f"workers={workers}"
    )

    store = RecordStore(records_path)
    aborted = threading.Event()
    lock = threading.Lock()
    started = time.time()
    counters = {"done": 0, "ok": 0, "err": 0, "streak": 0}

    def work(job: tuple[ds.Task, str]) -> None:
        task, arm = job
        if aborted.is_set():
            return
        try:
            record = evaluate_one(task, arm, cfg, out_dir, client)
        except Exception as exc:
            reason = batch_stopper(exc)
            if reason:
                if not aborted.is_set():
                    aborted.set()
                    print(
                        f"[abort] {reason} ({type(exc).__name__} at "
                        f"{task.task_id}/{arm}). Re-run with --resume.",
                        flush=True,
                    )
                return
            # The flags travel with the failure too: a flagged row that errors
            # belongs in `excluded`, not in the `errors` count of the headline
            # it was never going to be part of.
            _, rmeta = repair_image_question(task)
            record = {
                "task_id": task.task_id,
                "config": f"{cfg.name}:{arm}",
                "arm": arm,
                "eval_config": cfg.name,
                "model": cfg.model,
                "stem": task.stem,
                "qid": task.qid,
                "schema_type": task.row.get("schema_type"),
                "ok": False,
                "close": False,
                "error": f"{type(exc).__name__}: {exc}",
                **({} if arm != ARM_FIGURE else rmeta),
            }
        with lock:
            store.add(record)
            counters["done"] += 1
            counters["ok"] += bool(record.get("ok"))
            if record.get("error"):
                counters["err"] += 1
                counters["streak"] += 1
                if counters["streak"] >= cfg.max_consecutive_failures:
                    aborted.set()
                    print(
                        f"[abort] {counters['streak']} failures in a row with "
                        "nothing succeeding — treating this as an environment "
                        "outage, not as bad tasks",
                        flush=True,
                    )
            else:
                counters["streak"] = 0
            done, total = counters["done"], len(jobs)
            if done == 1 or done % 25 == 0:
                free = free_gb(out_dir)
                if free < cfg.min_free_gb:
                    aborted.set()
                    print(f"[abort] only {free:.1f} GB free", flush=True)
            eta = (
                f" eta={(total - done) * (time.time() - started) / done / 60:.0f}m"
                if done >= workers
                else ""
            )
            verdict = (
                f"ERROR {record['error']}"
                if record.get("error")
                else ("ok" if record["ok"] else "close" if record["close"] else "wrong")
            )
            print(
                f"[{done}/{total} ok={counters['ok']}{eta}] "
                f"{task.task_id:30s} {arm:13s} {verdict}",
                flush=True,
            )

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(work, jobs))
    finally:
        store.close()

    rows = read_records(records_path)
    print()
    print(summarise(rows, by_stem=args.by_stem))
    print(f"\n-> {records_path}")
    if aborted.is_set():
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
