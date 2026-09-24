"""Aggregate the V1 runs and read the H1 gate (AC-18, AC-19).

    python -m experiments.geostate.v1_report                 # the MLS runs
    python -m experiments.geostate.v1_report --root run/geostate/v1-smoke

Per arm and length bucket: success, target reached, wrong selection, moves,
tokens processed and generated, averaged over seeds. The gate compares the
arms on the long-construction bucket: Δsuccess = tokens − table, paired by
problem (each problem's per-arm success is its mean over seeds), with a
bootstrap 95% CI over problems; go / kill / reconsider per `V1_GATE`.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

from .client import write_json
from .configs import ARM_TABLE, ARM_TOKENS, ARTIFACT_ROOT, LENGTH_BUCKETS, V1, V1_GATE

FIELDS = (
    "success",
    "target_reached",
    "selection_error",
    "answered",
    "degenerate",
    "budget",
    "n_moves",
    "tokens_processed",
    "tokens_generated",
)


def load_runs(root: Path) -> dict[tuple[str, int], list[dict[str, Any]]]:
    """(arm, seed) -> episodes, from every `eval.jsonl` under root."""
    runs: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for path in sorted(root.glob("*/eval.jsonl")):
        train = path.parent / "train.json"
        if not train.exists():
            continue
        meta = json.loads(train.read_text())
        episodes = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
        runs[(meta["arm"], int(meta["seed"]))] = episodes
    return runs


def _select(episodes: list[dict[str, Any]], bucket: str) -> list[dict[str, Any]]:
    if bucket == "all":
        return episodes
    if bucket.startswith(">="):
        return [e for e in episodes if e["n_moves_reference"] >= V1_GATE.min_moves]
    return [e for e in episodes if e["bucket"] == bucket]


def _bootstrap(
    values: np.ndarray, rng: np.random.Generator, n: int = 1000
) -> tuple[float, float]:
    if values.size == 0:
        return math.nan, math.nan
    if values.size == 1:
        return float(values[0]), float(values[0])
    means = np.array(
        [rng.choice(values, size=values.size, replace=True).mean() for _ in range(n)]
    )
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def report(root: Path, seed: int = 0) -> tuple[dict[str, Any], str]:
    runs = load_runs(root)
    gate_bucket = f">={V1_GATE.min_moves} moves"
    buckets = [n for n, _, _ in LENGTH_BUCKETS] + ["all", gate_bucket]
    lines = [f"# V1 — state tokens vs matched-training baseline ({root})", ""]
    lines.append(
        "| arm | seeds | bucket | n | success | target reached | wrong selection | degenerate | budget | moves | tokens processed | tokens generated |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    summary: dict[str, dict[str, Any]] = {}
    per_problem: dict[str, dict[str, list[float]]] = {
        ARM_TOKENS: defaultdict(list),
        ARM_TABLE: defaultdict(list),
    }
    for arm in (ARM_TOKENS, ARM_TABLE):
        seeds = sorted(s for a, s in runs if a == arm)
        episodes = [e for s in seeds for e in runs[(arm, s)]]
        for e in episodes:
            per_problem[arm][e["problem_id"]].append(float(e["success"]))
        for bucket in buckets:
            sel = _select(episodes, bucket)
            if not sel:
                continue
            row = {
                f: float(
                    np.mean(
                        [
                            float(
                                e[f]
                                if f not in ("degenerate", "budget")
                                else e["error_kind"] == f
                            )
                            for e in sel
                        ]
                    )
                )
                for f in FIELDS
            }
            row["n"] = len(sel)
            row["seeds"] = seeds
            summary[f"{arm}/{bucket}"] = row
            lines.append(
                f"| {arm} | {len(seeds)} | {bucket} | {len(sel)} | {row['success']:.3f} | "
                f"{row['target_reached']:.3f} | {row['selection_error']:.3f} | {row['degenerate']:.3f} | "
                f"{row['budget']:.3f} | {row['n_moves']:.1f} | {row['tokens_processed']:.0f} | "
                f"{row['tokens_generated']:.0f} |"
            )
    # the gate: paired difference on the long bucket
    rng = np.random.default_rng(seed)
    gate: dict[str, Any] = {"verdict": None}
    tokens_eps = [
        e
        for s in sorted(s for a, s in runs if a == ARM_TOKENS)
        for e in runs[(ARM_TOKENS, s)]
    ]
    table_eps = [
        e
        for s in sorted(s for a, s in runs if a == ARM_TABLE)
        for e in runs[(ARM_TABLE, s)]
    ]
    long_ids = {e["problem_id"] for e in _select(tokens_eps, gate_bucket)} & {
        e["problem_id"] for e in _select(table_eps, gate_bucket)
    }
    deltas = np.array(
        [
            np.mean(per_problem[ARM_TOKENS][pid]) - np.mean(per_problem[ARM_TABLE][pid])
            for pid in sorted(long_ids)
        ]
    )
    lines.append("")
    if deltas.size == 0:
        gate["verdict"] = "no data"
        lines.append("Gate: no paired long-construction episodes for both arms yet.")
    else:
        lo, hi = _bootstrap(deltas, rng)
        delta = float(deltas.mean()) * 100.0
        gate.update(
            {
                "n_problems": int(deltas.size),
                "delta_pts": delta,
                "ci_pts": [lo * 100.0, hi * 100.0],
            }
        )
        if deltas.size < V1_GATE.n_eval:
            gate["verdict"] = "withheld"
            reason = f"only {deltas.size} paired problems, gate needs {V1_GATE.n_eval}"
        elif delta >= V1_GATE.go_delta_pts and lo > 0:
            gate["verdict"] = "GO"
            reason = "gain at/above the go margin with CI above zero"
        elif delta <= V1_GATE.kill_delta_pts:
            gate["verdict"] = "KILL"
            reason = "gain at/below the kill margin"
        else:
            gate["verdict"] = "RECONSIDER"
            reason = "between the kill and go margins"
        lines.append(
            f"Gate ({gate_bucket}, {deltas.size} paired problems): Δsuccess tokens − table = "
            f"{delta:+.1f} pts, CI95 [{lo * 100:+.1f}, {hi * 100:+.1f}] -> {gate['verdict']} ({reason}; "
            f"go ≥ {V1_GATE.go_delta_pts:.0f}, kill ≤ {V1_GATE.kill_delta_pts:.0f})."
        )
    text = "\n".join(lines) + "\n"
    return {
        "root": str(root),
        "runs": sorted(f"{a}/s{s}" for a, s in runs),
        "summary": summary,
        "gate": gate,
    }, text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path(ARTIFACT_ROOT) / "v1")
    ap.add_argument(
        "--out", type=Path, default=None, help="write report.{json,md} here"
    )
    args = ap.parse_args(argv)
    payload, text = report(args.root, seed=V1.eval_seed)
    out = args.out or args.root
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "report.json", payload)
    (out / "report.md").write_text(text)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
