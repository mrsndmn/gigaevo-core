"""Render every measured number as one markdown document.

The JSON artifacts live under ``results/``, which is gitignored, so this is
what actually preserves the study's numbers. ``--out`` writes the tracked
snapshot; with no arguments it prints to stdout.

    python -m experiments.vocab_factorization.report \
        --out experiments/vocab-factorization-tables.md

Benchmarks in columns, arms/experiments in rows.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

from experiments.vocab_factorization.configs import (
    EXPERIMENTS,
    HOST_MODEL,
    TRANSPLANTS,
)

SUITES = ["clean", "typo1", "typo5", "lower", "upper", "title", "randcase"]
CTX_SUITES = ["clean", "typo1", "typo5", "typo10", "randcase", "mismatch"]


def _fmt(v, nd=4):
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "**NO**"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def _table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join("---" for _ in headers) + "|")
    for r in rows:
        out.append("| " + " | ".join(_fmt(v) for v in r) + " |")
    return "\n".join(out)


def _load(runs_dir: Path, order: list[str]) -> tuple[list[str], dict]:
    found = {}
    for p in runs_dir.glob("*.json"):
        d = json.loads(p.read_text())
        found[d["experiment"]] = d
    names = [n for n in order if n in found] + [
        n for n in sorted(found) if n not in order
    ]
    return names, found


# ---------------------------------------------------------------------------


def screen_table(path: Path) -> str:
    rows_in = json.loads(path.read_text())
    head = [
        "arm", "vocab", "tok/byte", "pieces/word", "whole-word %",
        "tpb UPPER", "tpb Title", "tpb randcase", "tpb typo5",
        "embed params (d=512)", "lossless",
    ]
    rows = [
        [
            r["arm"], r["vocab"], r["tpb.clean"], round(r["pieces_per_word"], 3),
            f"{100 * r['whole_word_rate']:.1f}", r["tpb.upper"], r["tpb.title"],
            r["tpb.randcase"], r["tpb.typo5"],
            r["embed_params_d512_tied"], r["roundtrip"],
        ]
        for r in rows_in
    ]
    return _table(head, rows)


def bpb_tables(runs_dir: Path) -> str:
    names, found = _load(runs_dir, [e.name for e in EXPERIMENTS])
    if not names:
        return ""
    abs_rows = [
        [n] + [found[n][f"eval.{s}"]["bpb"] for s in SUITES] for n in names
    ]
    delta_rows = [
        [n] + [found[n][f"delta.{s}"] for s in SUITES[1:]] for n in names
    ]
    head_rows = []
    for n in names:
        e = found[n]["eval.clean"]
        heads = {k.split(".", 1)[1]: v for k, v in e.items() if k.startswith("bpb.")}
        head_rows.append(
            [n, e["bpb"], heads.get("stem"), heads.get("role"), heads.get("suffix")]
        )
    return "\n\n".join(
        [
            "### Absolute bits per byte",
            _table(["experiment", *[f"BPB {s}" for s in SUITES]], abs_rows),
            "### Delta vs clean",
            _table(["experiment", *[f"d {s}" for s in SUITES[1:]]], delta_rows),
            "### Clean BPB split by prediction head",
            _table(["experiment", "BPB total", "stem", "role", "suffix"], head_rows),
        ]
    )


def cost_table(runs_dir: Path) -> str:
    names, found = _load(runs_dir, [e.name for e in EXPERIMENTS])
    if not names:
        return ""
    head = [
        "experiment", "BPB clean", "total params", "embed params", "tok/byte",
        "context bytes", "steps", "train tokens", "minutes",
    ]
    rows = []
    for n in names:
        d = found[n]
        p = d["params"]
        rows.append([
            n, d["eval.clean"]["bpb"], p["total"],
            p["embedding"] + p["output_heads"] + p["ngram"],
            d["tokens_per_byte_train"], round(d["context_bytes"]),
            d["steps"], d["train_tokens"], f"{d['train_seconds'] / 60:.1f}",
        ])
    return _table(head, rows)


def context_table(runs_dir: Path) -> str:
    """Corrupted-context BPB; only arms re-run after the metric was added."""
    names, found = _load(runs_dir, [e.name for e in EXPERIMENTS])
    names = [n for n in names if "ctx.clean" in found[n]]
    if not names:
        return ""
    head = [
        "experiment", "ctx BPB clean",
        *[f"d {s}" for s in CTX_SUITES[1:]],
        "typo10 / mismatch",
    ]
    rows = []
    for n in names:
        d = found[n]
        deltas = [d.get(f"ctxdelta.{s}") for s in CTX_SUITES[1:]]
        mm = d.get("ctxdelta.mismatch")
        t10 = d.get("ctxdelta.typo10")
        ratio = f"{t10 / mm:.1%}" if mm else "-"
        rows.append([n, d["ctx.clean"]["bpb"], *deltas, ratio])
    return _table(head, rows)


def transplant_table(runs_dir: Path) -> str:
    names, found = _load(runs_dir, [t.name for t in TRANSPLANTS])
    if not names:
        return ""
    suites = ["typo1", "typo5", "lower", "upper", "title", "randcase"]
    head = [
        "experiment", "init", "BPB step0", "BPB final",
        *[f"d {s}" for s in suites],
        "vocab", "embed params", "total params", "steps", "minutes",
    ]
    rows = []
    for n in names:
        d = found[n]
        p = d["params"]
        rows.append([
            n, d["init"], d["step0_clean_bpb"], d["eval.clean"]["bpb"],
            *[d[f"delta.{s}"] for s in suites],
            d["heads"]["stem"], p["embedding"] + p["output_heads"], p["total"],
            d["steps"], f"{d['train_seconds'] / 60:.1f}",
        ])
    return _table(head, rows)


# ---------------------------------------------------------------------------


def build_document(args) -> str:
    parts = [
        "# Vocabulary factorization — measured numbers",
        "",
        "Generated snapshot of every metric in the study. The underlying JSON "
        "lives under `results/`, which is gitignored; this file is the tracked "
        "record. Interpretation and conclusions live in "
        "`experiments/vocab-factorization.md` — this file is data only.",
        "",
        f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"- Transplant host: `{HOST_MODEL}`",
        "- Corpus: ru+en Wikipedia, NFC, document-hash split (`data.py`)",
        "- Regenerate: `python -m experiments.vocab_factorization.report "
        "--out experiments/vocab-factorization-tables.md`",
        "",
        "Measured run-to-run noise (bpe8k trained twice, identical config): "
        "**0.0016 BPB**. Treat smaller differences as nothing.",
        "",
    ]

    if args.screen.exists():
        parts += [
            "## 1. Training-free screen",
            "",
            "Absolute tokens/byte per suite, *not* the inflation ratio: "
            "inflation is relative to each arm's own clean baseline and so "
            "rewards a bad baseline (it ranked `fact8k-nosuf` best on randcase "
            "purely because its clean tokens/byte was worst).",
            "",
            screen_table(args.screen),
            "",
        ]

    if args.runs.exists() and any(args.runs.glob("*.json")):
        parts += ["## 2. From-scratch 33M grid", "", bpb_tables(args.runs), ""]
        parts += ["### Cost and capacity", "", cost_table(args.runs), ""]
        ctx = context_table(args.runs)
        if ctx:
            parts += [
                "## 3. Corrupted-context BPB",
                "",
                "Corruption confined to unscored context; the denominator is "
                "clean-suffix bytes only, so nothing is charged for emitting "
                "noise. `mismatch` replaces the context with an unrelated "
                "document and calibrates how much any context damage can cost.",
                "",
                ctx,
                "",
            ]

    if args.transplant.exists() and any(args.transplant.glob("*.json")):
        parts += [
            "## 4. Transplant onto a frozen Qwen3-1.7B",
            "",
            "Body frozen; only the embedding table and heads train.",
            "",
            transplant_table(args.transplant),
            "",
        ]

    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", type=Path, default=Path("results/screen2.json"))
    ap.add_argument("--runs", type=Path, default=Path("results/runs"))
    ap.add_argument("--transplant", type=Path, default=Path("results/transplant"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    doc = build_document(args)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(doc)
        print(f"wrote {args.out} ({len(doc.splitlines())} lines)")
    else:
        print(doc)


if __name__ == "__main__":
    main()
