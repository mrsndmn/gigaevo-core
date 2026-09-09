"""Measure the endpoint's throughput at several concurrency levels.

Picks `DEFAULT_WORKERS` with data instead of a guess. The server is a shared
vLLM instance: it batches, so per-request latency grows with concurrency while
total throughput keeps climbing until the batch saturates. What we want is the
knee — past it we are queueing behind ourselves and crowding other users for
almost nothing.

    python -m experiments.noregeo_synth.probe_concurrency --levels 1,4,8,16

Real prompts from real tasks (distinct ones per request, so nothing is served
from a prefix cache), and the generated JSON is thrown away.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import statistics as st
import time

from . import dataset as ds
from .canvas import build_preamble
from .configs import CONFIGS_BY_NAME, DEFAULT_CONFIG
from .generate import _call, _client
from .prompts import build_messages, response_schema


def _one(client, cfg, task) -> tuple[float, int, str | None]:
    preamble = build_preamble(task.row, cfg)
    messages = build_messages(task.row, cfg, preamble)
    schema = response_schema(task.row, cfg)
    t0 = time.time()
    try:
        _, _, usage = _call(client, cfg, messages, schema)
    except Exception as exc:
        return time.time() - t0, 0, f"{type(exc).__name__}: {exc}"[:120]
    return time.time() - t0, int(usage.get("completion_tokens", 0)), None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=DEFAULT_CONFIG, choices=list(CONFIGS_BY_NAME))
    ap.add_argument("--levels", default="1,4,8,16")
    ap.add_argument(
        "--waves",
        type=int,
        default=2,
        help="requests per level = workers * waves (>=2 to reach steady state)",
    )
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    cfg = CONFIGS_BY_NAME[args.config]
    levels = [int(x) for x in args.levels.split(",")]
    tasks = ds.load_tasks()
    import random

    pool = random.Random(args.seed).sample(tasks, sum(levels) * args.waves)
    client, _ = _client(cfg)

    print(f"model={cfg.model}, {args.waves} waves per level")
    print(
        f"{'workers':>7} {'req/min':>8} {'lat p50':>8} {'lat p90':>8} "
        f"{'tok/s':>7} {'wall':>7} errors"
    )
    cursor = 0
    for k in levels:
        count = k * args.waves
        batch = pool[cursor : cursor + count]
        cursor += count
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=k) as ex:
            results = list(ex.map(lambda t: _one(client, cfg, t), batch))
        wall = time.time() - t0
        lat = sorted(r[0] for r in results)
        toks = sum(r[1] for r in results)
        errs = [r[2] for r in results if r[2]]
        # Little's law: with k requests in flight and a mean service time of
        # `mean(lat)`, steady-state throughput is k / mean(lat). Dividing the
        # batch size by its makespan instead would measure E[max latency] over
        # a heavy tail, which falls as k grows even at constant throughput.
        thr = k / st.mean(lat) * 60
        print(
            f"{k:>7} {thr:>8.1f} {lat[len(lat) // 2]:>7.1f}s "
            f"{lat[int(0.9 * (len(lat) - 1))]:>7.1f}s {toks / wall:>7.1f} "
            f"{wall:>6.1f}s {len(errs)}" + (f"  {errs[0]}" if errs else "")
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
