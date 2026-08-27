"""Render the screen and training results as markdown tables.

Benchmarks in columns, arms/experiments in rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.vocab_factorization.configs import EXPERIMENTS


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


def screen_table(path: Path) -> str:
    rows_in = json.loads(path.read_text())
    headers = [
        "arm",
        "vocab",
        "tok/byte",
        "pieces/word",
        "whole-word %",
        "infl UPPER",
        "infl Title",
        "infl randcase",
        "infl typo5",
        "embed params (d=512)",
        "lossless",
    ]
    rows = []
    for r in rows_in:
        rows.append(
            [
                r["arm"],
                r["vocab"],
                r["tpb.clean"],
                r["pieces_per_word"],
                100 * r["whole_word_rate"],
                r["infl.upper"],
                r["infl.title"],
                r["infl.randcase"],
                r["infl.typo5"],
                r["embed_params_d512_tied"],
                r["roundtrip"],
            ]
        )
    return _table(headers, rows)


def runs_table(runs_dir: Path) -> tuple[str, str]:
    order = [e.name for e in EXPERIMENTS]
    found = {}
    for p in runs_dir.glob("*.json"):
        d = json.loads(p.read_text())
        found[d["experiment"]] = d
    names = [n for n in order if n in found] + [n for n in found if n not in order]

    suites = ["clean", "typo1", "typo5", "lower", "upper", "title", "randcase"]
    abs_headers = ["experiment", "BPB clean"] + [f"BPB {s}" for s in suites[1:]]
    abs_rows, delta_rows = [], []
    for n in names:
        d = found[n]
        abs_rows.append([n] + [d[f"eval.{s}"]["bpb"] for s in suites])
        delta_rows.append([n] + [d[f"delta.{s}"] for s in suites[1:]])

    cost_headers = [
        "experiment",
        "BPB clean",
        "total params",
        "embed params",
        "tok/byte",
        "context bytes",
        "steps",
        "train tokens",
        "minutes",
    ]
    cost_rows = []
    for n in names:
        d = found[n]
        cost_rows.append(
            [
                n,
                d["eval.clean"]["bpb"],
                d["params"]["total"],
                d["params"]["embedding"]
                + d["params"]["output_heads"]
                + d["params"]["ngram"],
                d["tokens_per_byte_train"],
                round(d["context_bytes"]),
                d["steps"],
                d["train_tokens"],
                round(d["train_seconds"] / 60, 1),
            ]
        )

    delta_headers = ["experiment"] + [f"d {s}" for s in suites[1:]]
    return (
        _table(abs_headers, abs_rows) + "\n\n" + _table(delta_headers, delta_rows),
        _table(cost_headers, cost_rows),
    )


def transplant_table(runs_dir: Path) -> str:
    from experiments.vocab_factorization.configs import TRANSPLANTS

    order = [t.name for t in TRANSPLANTS]
    found = {}
    for p in runs_dir.glob("*.json"):
        d = json.loads(p.read_text())
        found[d["experiment"]] = d
    names = [n for n in order if n in found] + [n for n in found if n not in order]
    if not names:
        return ""

    suites = ["typo1", "typo5", "lower", "upper", "title", "randcase"]
    head = [
        "experiment",
        "init",
        "BPB step0",
        "BPB final",
        *[f"d {s}" for s in suites],
        "vocab",
        "embed params",
        "total params",
        "steps",
        "minutes",
    ]
    rows = []
    for n in names:
        d = found[n]
        p = d["params"]
        rows.append(
            [
                n,
                d["init"],
                d["step0_clean_bpb"],
                d["eval.clean"]["bpb"],
                *[d[f"delta.{s}"] for s in suites],
                d["heads"]["stem"],
                p["embedding"] + p["output_heads"],
                p["total"],
                d["steps"],
                round(d["train_seconds"] / 60, 1),
            ]
        )
    return _table(head, rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", type=Path, default=Path("results/screen.json"))
    ap.add_argument("--runs", type=Path, default=Path("results/runs"))
    ap.add_argument("--transplant", type=Path, default=Path("results/transplant"))
    args = ap.parse_args()

    if args.transplant.exists() and any(args.transplant.glob("*.json")):
        print("## Transplant onto frozen Qwen3-1.7B\n")
        print(transplant_table(args.transplant))
        print()

    if args.screen.exists():
        print("## Training-free screen\n")
        print(screen_table(args.screen))
        print()
    if args.runs.exists() and any(args.runs.glob("*.json")):
        bpb, cost = runs_table(args.runs)
        print("\n## Bits per byte\n")
        print(bpb)
        print("\n## Cost and capacity\n")
        print(cost)


if __name__ == "__main__":
    main()
