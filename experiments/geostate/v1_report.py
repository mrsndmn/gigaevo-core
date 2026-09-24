"""Aggregate the V1 runs and read the H1 gate (AC-18, AC-19).

    python -m experiments.geostate.v1_report                        # current protocol
    python -m experiments.geostate.v1_report --protocol f6181a88e7c2 # an earlier one
    python -m experiments.geostate.v1_report --root run/geostate/v1-smoke

Per arm and length bucket: success, target reached, wrong selection, moves,
tokens — each problem's value is its mean over seeds, and every rate carries a
bootstrap 95% CI over problems. The gate compares the arms on the
long-construction bucket: Δsuccess = tokens − table, paired by problem, with a
bootstrap CI; go / kill / reconsider per `V1_GATE`, and *withheld* unless every
configured arm × seed is present and its evaluation is complete.

Runs are read from `<root>/<arm>-s<seed>-<protocol>` (the launcher's layout);
directories named `<arm>-s<seed>` or `<arm>` (smoke layout) are accepted as
unversioned. Two directories for one arm × seed is an error, never a silent
replacement.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np

from .client import read_json, write_json
from .configs import (
    ARM_TABLE,
    ARM_TOKENS,
    ARTIFACT_ROOT,
    LENGTH_BUCKETS,
    V1,
    V1_GATE,
    V1Config,
)

RUN_DIR_RE = re.compile(
    r"^(?P<arm>tokens|table)(?:-s(?P<seed>\d+))?(?:-(?P<protocol>[0-9a-f]{12}))?$"
)
RATES = ("success", "target_reached", "selection_error", "answered")
MEANS = ("n_moves", "tokens_processed", "tokens_generated")


def current_protocol(cfg: V1Config) -> str:
    from .run_v1_jobs import protocol_hash_v1

    return protocol_hash_v1(cfg)


def load_runs(
    root: Path, cfg: V1Config = V1, protocol: str | None = None
) -> dict[tuple[str, int], dict[str, Any]]:
    """(arm, seed) -> {"episodes", "complete", "dir", ...} for the wanted protocol."""
    wanted = protocol or current_protocol(cfg)
    runs: dict[tuple[str, int], dict[str, Any]] = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        m = RUN_DIR_RE.match(d.name)
        if not m:
            continue
        if m.group("protocol") and m.group("protocol") != wanted:
            continue
        eval_path = d / "eval.jsonl"
        if not eval_path.exists():
            continue
        train = read_json(d / "train.json") if (d / "train.json").exists() else {}
        arm = train.get("arm", m.group("arm"))
        seed = int(train.get("seed", m.group("seed") or 0))
        if (arm, seed) in runs:
            raise RuntimeError(
                f"two runs for {arm} seed {seed}: {runs[(arm, seed)]['dir']} and {d}"
            )
        episodes = []
        for line in eval_path.read_text().splitlines():
            if line.strip():
                try:
                    episodes.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        metrics = read_json(d / "metrics.json") if (d / "metrics.json").exists() else {}
        manifest = read_json(d / "run.json") if (d / "run.json").exists() else {}
        expected = manifest.get("n_eval", metrics.get("n_expected"))
        complete = (
            bool(metrics) and expected is not None and len(episodes) >= int(expected)
        )
        runs[(arm, seed)] = {
            "dir": str(d),
            "episodes": episodes,
            "complete": complete,
            "n_expected": expected,
            "protocol": m.group("protocol"),
        }
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


def _per_problem(episodes: list[dict[str, Any]], field: str) -> dict[str, float]:
    """Mean of `field` per problem over seeds."""
    acc: dict[str, list[float]] = defaultdict(list)
    for e in episodes:
        acc[e["problem_id"]].append(float(e[field]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def report(
    root: Path, cfg: V1Config = V1, protocol: str | None = None, seed: int = 0
) -> tuple[dict[str, Any], str]:
    runs = load_runs(root, cfg, protocol)
    rng = np.random.default_rng(seed)
    gate_bucket = f">={V1_GATE.min_moves} moves"
    buckets = [n for n, _, _ in LENGTH_BUCKETS] + ["all", gate_bucket]
    lines = [f"# V1 — state tokens vs matched-training baseline ({root})", ""]
    lines.append(
        "| arm | seeds | bucket | problems | success [CI95] | target reached | "
        "wrong selection [CI95] | moves | tokens processed [CI95] | tokens generated |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    summary: dict[str, dict[str, Any]] = {}
    for arm in (ARM_TOKENS, ARM_TABLE):
        seeds = sorted(s for a, s in runs if a == arm)
        episodes = [e for s in seeds for e in runs[(arm, s)]["episodes"]]
        for bucket in buckets:
            sel = _select(episodes, bucket)
            if not sel:
                continue
            row: dict[str, Any] = {
                "n_problems": len({e["problem_id"] for e in sel}),
                "seeds": seeds,
            }
            for f in RATES + MEANS:
                values = np.array(list(_per_problem(sel, f).values()))
                row[f] = float(values.mean())
                row[f + "_ci"] = list(_bootstrap(values, rng))
            for kind in ("degenerate", "budget"):
                row[kind] = float(np.mean([e.get("error_kind") == kind for e in sel]))
            summary[f"{arm}/{bucket}"] = row
            sc, ec, tc = (
                row["success_ci"],
                row["selection_error_ci"],
                row["tokens_processed_ci"],
            )
            lines.append(
                f"| {arm} | {len(seeds)} | {bucket} | {row['n_problems']} | "
                f"{row['success']:.3f} [{sc[0]:.3f}, {sc[1]:.3f}] | "
                f"{row['target_reached']:.3f} | "
                f"{row['selection_error']:.3f} [{ec[0]:.3f}, {ec[1]:.3f}] | "
                f"{row['n_moves']:.1f} | {row['tokens_processed']:.0f} "
                f"[{tc[0]:.0f}, {tc[1]:.0f}] | {row['tokens_generated']:.0f} |"
            )
    # completeness: every configured arm x seed, fully evaluated
    missing = [f"{a}/s{s}" for a in cfg.arms for s in cfg.seeds if (a, s) not in runs]
    incomplete = [
        f"{a}/s{s} ({len(r['episodes'])}/{r['n_expected']})"
        for (a, s), r in sorted(runs.items())
        if not r["complete"]
    ]
    # the gate: paired difference on the long bucket
    gate: dict[str, Any] = {
        "verdict": None,
        "missing": missing,
        "incomplete": incomplete,
    }
    tokens_eps = [
        e for (a, _), r in runs.items() if a == ARM_TOKENS for e in r["episodes"]
    ]
    table_eps = [
        e for (a, _), r in runs.items() if a == ARM_TABLE for e in r["episodes"]
    ]
    tok_pp = _per_problem(_select(tokens_eps, gate_bucket), "success")
    tab_pp = _per_problem(_select(table_eps, gate_bucket), "success")
    paired = sorted(set(tok_pp) & set(tab_pp))
    deltas = np.array([tok_pp[p] - tab_pp[p] for p in paired])
    lines.append("")
    if deltas.size:
        lo, hi = _bootstrap(deltas, rng)
        delta = float(deltas.mean()) * 100.0
        gate.update(
            {
                "n_problems": int(deltas.size),
                "delta_pts": delta,
                "ci_pts": [lo * 100.0, hi * 100.0],
            }
        )
        lines.append(
            f"Paired Δsuccess tokens − table on {gate_bucket} ({deltas.size} problems): "
            f"{delta:+.1f} pts, CI95 [{lo * 100:+.1f}, {hi * 100:+.1f}]."
        )
    if missing or incomplete:
        gate["verdict"] = "withheld"
        why = []
        if missing:
            why.append("missing runs: " + ", ".join(missing))
        if incomplete:
            why.append("incomplete runs: " + ", ".join(incomplete))
        lines.append("Gate verdict: withheld — " + "; ".join(why) + ".")
    elif deltas.size == 0:
        gate["verdict"] = "no data"
        lines.append("Gate: no paired long-construction episodes for both arms.")
    elif deltas.size < V1_GATE.n_eval:
        gate["verdict"] = "withheld"
        lines.append(
            f"Gate verdict: withheld — {deltas.size} paired problems, the gate needs "
            f"{V1_GATE.n_eval}."
        )
    else:
        lo_pts = gate["ci_pts"][0]
        delta = gate["delta_pts"]
        if delta >= V1_GATE.go_delta_pts and lo_pts > 0:
            gate["verdict"], reason = (
                "GO",
                "gain at/above the go margin with CI above zero",
            )
        elif delta <= V1_GATE.kill_delta_pts:
            gate["verdict"], reason = "KILL", "gain at/below the kill margin"
        else:
            gate["verdict"], reason = "RECONSIDER", "between the kill and go margins"
        lines.append(
            f"Gate verdict: {gate['verdict']} ({reason}; go ≥ {V1_GATE.go_delta_pts:.0f} pts, "
            f"kill ≤ {V1_GATE.kill_delta_pts:.0f} pts)."
        )
    text = "\n".join(lines) + "\n"
    payload = {
        "root": str(root),
        "protocol": protocol or current_protocol(cfg),
        "runs": {
            f"{a}/s{s}": {k: v for k, v in r.items() if k != "episodes"}
            for (a, s), r in sorted(runs.items())
        },
        "summary": summary,
        "gate": gate,
    }
    return payload, text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path(ARTIFACT_ROOT) / "v1")
    ap.add_argument(
        "--protocol", default=None, help="protocol hash to report (default: current)"
    )
    ap.add_argument(
        "--out", type=Path, default=None, help="write report.{json,md} here"
    )
    args = ap.parse_args(argv)
    payload, text = report(args.root, V1, args.protocol, seed=V1.eval_seed)
    out = args.out or args.root
    out.mkdir(parents=True, exist_ok=True)
    suffix = f"-{payload['protocol']}"
    write_json(out / f"report{suffix}.json", payload)
    (out / f"report{suffix}.md").write_text(text)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
