"""Generate the synthetic "reasoning + incremental plot" dataset from NoReGeo.

For each benchmark task the model returns a JSON solution: N reasoning steps,
each with a matplotlib fragment that adds that step's construction to a shared
canvas. We execute the fragments in order (one PNG per step), then keep only
samples where the code ran, every step changed the picture, and the final
answer equals the benchmark's golden answer.

Usage (single-sample smoke run):

    python -m experiments.noregeo_synth.generate --task outer_circle_center:6200

    python -m experiments.noregeo_synth.generate \\
        --config qwen38-27b --n 20 --stem triangle_area --out run/noregeo_synth

Everything about *what* is generated lives in `configs.py`.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any

from . import dataset as ds
from .canvas import build_preamble
from .configs import (
    CONFIGS_BY_NAME,
    DEFAULT_CONFIG,
    OUT_DIR,
    SMOKE_TASK,
    GenConfig,
)
from .credentials import resolve
from .execute import run_steps
from .prompts import build_messages, repair_messages, response_schema
from .verify import answer_matches, contact_sheet, pixel_deltas


def _client(cfg: GenConfig):
    from openai import OpenAI

    creds = resolve()
    base_url = cfg.base_url or creds.base_url
    days = creds.expires_in_days
    if days is not None and days < 3:
        print(f"[warn] inference token expires in {days:.1f} days", file=sys.stderr)
    return OpenAI(
        base_url=base_url, api_key=creds.api_key, timeout=cfg.request_timeout
    ), creds


def _call(client, cfg: GenConfig, messages: list[dict], schema: dict):
    kwargs: dict[str, Any] = dict(
        model=cfg.model,
        messages=messages,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_tokens=cfg.max_tokens,
        response_format={"type": "json_schema", "json_schema": schema},
    )
    if cfg.seed is not None:
        kwargs["seed"] = cfg.seed
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
    if not msg.content:
        raise RuntimeError(
            f"empty completion (finish_reason={choice.finish_reason}, "
            f"thinking chars={len(thinking or '')}). Raise `max_tokens` or set "
            "`thinking='off'` in the config."
        )
    return json.loads(msg.content), thinking, usage


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
    """Every acceptance signal for one candidate solution, plus what failed."""
    problems: list[str] = []

    if not exec_res.ok:
        failed = exec_res.first_failure
        detail = exec_res.error or (failed.error if failed else "unknown failure")
        if failed is not None and failed.index <= len(solution["steps"]):
            code = solution["steps"][failed.index - 1]["code"]
            detail = f"Step {failed.index} code:\n```python\n{code}\n```\n\n{detail}"
        problems.append(f"The code did not run:\n{detail}")

    images = [s.image for s in exec_res.steps if s.ok and s.image]
    deltas = pixel_deltas(images) if images else []
    dead = [i for i, d in enumerate(deltas, 1) if not d or d < cfg.min_pixel_delta]
    if dead:
        problems.append(
            f"Step(s) {dead} drew nothing new — every step must add something "
            "visible to the canvas."
        )

    ans_ok, ans_detail = answer_matches(row, solution.get("final_answer"), cfg)
    if exec_res.ok and not ans_ok:
        problems.append(f"`final_answer` is wrong: {ans_detail}.")

    code_ok, code_detail = True, "not checked"
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

    drawn_ok, draw_dist = True, exec_res.answer_point_distance
    if exec_res.ok and row.get("schema_type") == "point":
        drawn_ok = draw_dist is not None and draw_dist <= cfg.draw_atol
        if not drawn_ok:
            where = (
                "no point is marked at all"
                if draw_dist is None
                else (f"the nearest marked point is {draw_dist:.3g} away")
            )
            problems.append(
                f"The answer is not drawn where it belongs: {where}. "
                "Mark the answer point at its true coordinates."
            )

    return {
        "deltas": deltas,
        "incremental_ok": bool(deltas) and not dead,
        "answer_ok": ans_ok,
        "answer_detail": ans_detail,
        "code_answer_ok": code_ok,
        "code_answer_detail": code_detail,
        "drawn_ok": drawn_ok,
        "draw_distance": draw_dist,
        "problems": problems,
    }


def _feedback(solution: dict, exec_res, checks: dict) -> str:
    return "\n\n".join(checks["problems"])


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
    solution, thinking, usage = _call(client, cfg, messages, schema)
    exec_res = run_steps(preamble, solution["steps"], sample_dir, cfg, target)
    checks = run_checks(row, solution, exec_res, cfg)

    rounds = 0
    while checks["problems"] and rounds < cfg.max_repair_rounds:
        rounds += 1
        solution, thinking, usage2 = _call(
            client,
            cfg,
            repair_messages(messages, solution, _feedback(solution, exec_res, checks)),
            schema,
        )
        for k, v in usage2.items():
            if isinstance(v, (int, float)) and isinstance(usage.get(k), (int, float)):
                usage[k] += v
        exec_res = run_steps(preamble, solution["steps"], sample_dir, cfg, target)
        checks = run_checks(row, solution, exec_res, cfg)

    images = [s.image for s in exec_res.steps if s.ok and s.image]
    deltas = checks["deltas"]

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
        "exec_ok": exec_res.ok,
        "exec_error": exec_res.error,
        "incremental_ok": checks["incremental_ok"],
        "accepted": not checks["problems"],
        "problems": checks["problems"],
        "repair_rounds": rounds,
        "thinking": thinking,
        "usage": usage,
        "wall_time_s": round(time.time() - t0, 1),
        "preamble": "preamble.py",
        "reference_image": reference,
        "sample_dir": str(sample_dir),
    }
    (sample_dir / "record.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2)
    )
    return record


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
    ap.add_argument("--seed", type=int, default=0, help="sampling seed")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--model", help="override the config's model id")
    ap.add_argument("--no-reveal-answer", action="store_true")
    args = ap.parse_args(argv)

    cfg = CONFIGS_BY_NAME[args.config]
    overrides = {}
    if args.model:
        overrides["model"] = args.model
    if args.no_reveal_answer:
        overrides["reveal_answer"] = False
    if overrides:
        cfg = GenConfig(**{**asdict(cfg), **overrides})

    tasks = ds.load_tasks()
    selected: list[ds.Task] = []
    if args.n:
        pool = [t for t in tasks if not args.stem or t.stem == args.stem]
        selected = random.Random(args.seed).sample(pool, min(args.n, len(pool)))
    for spec in args.task or ([] if args.n else [SMOKE_TASK]):
        selected.append(ds.select(tasks, spec))

    client, creds = _client(cfg)
    print(
        f"model={cfg.model} endpoint={cfg.base_url} creds={creds.source} "
        f"samples={len(selected)}"
    )

    records_path = args.out / cfg.name / "records.jsonl"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    accepted = 0
    for task in selected:
        rec = generate_one(task, cfg, args.out, client=client)
        accepted += rec["accepted"]
        with records_path.open("a") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        flags = " ".join(
            f"{k}={'ok' if rec[k] else 'FAIL'}"
            for k in (
                "exec_ok",
                "incremental_ok",
                "answer_ok",
                "code_answer_ok",
                "drawn_ok",
            )
        )
        flags += f" [{rec['answer_detail']}]"
        print(
            f"  {task.task_id:34s} steps={len(rec['steps'])} "
            f"repairs={rec['repair_rounds']} {flags} -> {rec['sample_dir']}"
        )
    print(f"accepted {accepted}/{len(selected)} -> {records_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
