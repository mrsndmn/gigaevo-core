"""Local API cohorts paired with frozen short text problems (no MLS submission).

    python -m experiments.geostate.run_figure_short50 --dry
    python -u -m experiments.geostate.run_figure_short50
    python -u -m experiments.geostate.run_figure_short50 --config e0-synthetic-figure-short-next50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .client import write_json
from .configs import (
    ARTIFACT_ROOT,
    FIGURE_SHORT50_BUCKET,
    FIGURE_SHORT50_N,
    FIGURE_SHORT50_OFFSETS,
    FIGURE_SHORT50_SOURCE,
    config_named,
)
from .consistency import (
    Copy, figure_for, fingerprint, job_dir, prepare, protocol_hash, run, summarize,
)
from .generator import load_problems, save_problems


def freeze(out: Path, config_name: str = "e0-synthetic-figure-short50") -> None:
    cfg = config_named(config_name)
    offset = FIGURE_SHORT50_OFFSETS[config_name]
    source = Path(FIGURE_SHORT50_SOURCE)
    problems = [
        p for p in load_problems(source / "problems.jsonl")
        if p.bucket == FIGURE_SHORT50_BUCKET
    ][offset:offset + FIGURE_SHORT50_N]
    if len(problems) != FIGURE_SHORT50_N:
        raise ValueError("not enough short problems in the frozen text source")
    ids = {p.id for p in problems}
    copies = [
        json.loads(line)
        for line in (source / "copies.jsonl").read_text().splitlines()
        if line.strip() and json.loads(line)["problem_id"] in ids
    ]
    if any(sum(c["problem_id"] == p.id for c in copies) != cfg.copies for p in problems):
        raise ValueError("each selected problem must have all eight frozen copies")
    if (out / "prepare.json").exists():
        manifest = json.loads((out / "prepare.json").read_text())
        if manifest.get("fingerprint") != fingerprint(cfg):
            raise ValueError("existing pilot has a different fingerprint")
        frozen_problems = load_problems(out / "problems.jsonl")
        frozen_copies = [json.loads(line) for line in (out / "copies.jsonl").read_text().splitlines() if line.strip()]
        prepared = [(p, [Copy.from_dict(c["copy"]) for c in frozen_copies if c["problem_id"] == p.id]) for p in frozen_problems]
        expected = [(p.to_dict(), [c["copy"] for c in copies if c["problem_id"] == p.id]) for p in problems]
        actual = [(p.to_dict(), [c.to_dict() for c in cs]) for p, cs in prepared]
        if actual != expected:
            raise ValueError("existing pilot differs from the paired text subset")
        return
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing to overwrite a partially prepared pilot")
    out.mkdir(parents=True, exist_ok=True)
    save_problems(problems, out / "problems.jsonl")
    (out / "copies.jsonl").write_text("".join(json.dumps(c) + "\n" for c in copies))
    write_json(out / "prepare.json", {
        "config": cfg.name,
        "fingerprint": fingerprint(cfg),
        "n_problems": len(problems),
        "n_copies": len(copies),
        "paired_text_source": str(source),
    })


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--config", choices=tuple(FIGURE_SHORT50_OFFSETS), default="e0-synthetic-figure-short50")
    args = ap.parse_args()
    cfg = config_named(args.config)
    out = Path(ARTIFACT_ROOT) / f"{cfg.name}-{protocol_hash(cfg)}"
    freeze(out, cfg.name)
    print(f"Pilot: {FIGURE_SHORT50_N} short problems, 400 requests, {cfg.workers} workers; {out}", flush=True)
    if args.dry:
        return 0
    # Matplotlib is not thread-safe; render before the concurrent API loop.
    for problem, copies in prepare(cfg, out):
        for copy in copies:
            figure_for(copy, job_dir(out, cfg.models[0], problem, copy))
    print("All 400 figures ready; starting API requests", flush=True)
    code = run(cfg, out)
    payload, report = summarize(cfg, out)
    write_json(out / "summary.json", payload)
    (out / "summary.md").write_text(report)
    print(report, flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
