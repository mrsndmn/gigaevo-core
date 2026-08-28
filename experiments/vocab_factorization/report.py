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
    DISTILLS,
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
        "arm",
        "vocab",
        "tok/byte",
        "pieces/word",
        "whole-word %",
        "tpb UPPER",
        "tpb Title",
        "tpb randcase",
        "tpb typo5",
        "embed params (d=512)",
        "lossless",
    ]
    rows = [
        [
            r["arm"],
            r["vocab"],
            r["tpb.clean"],
            round(r["pieces_per_word"], 3),
            f"{100 * r['whole_word_rate']:.1f}",
            r["tpb.upper"],
            r["tpb.title"],
            r["tpb.randcase"],
            r["tpb.typo5"],
            r["embed_params_d512_tied"],
            r["roundtrip"],
        ]
        for r in rows_in
    ]
    return _table(head, rows)


def bpb_tables(runs_dir: Path) -> str:
    names, found = _load(runs_dir, [e.name for e in EXPERIMENTS])
    if not names:
        return ""
    abs_rows = [[n] + [found[n][f"eval.{s}"]["bpb"] for s in SUITES] for n in names]
    delta_rows = [[n] + [found[n][f"delta.{s}"] for s in SUITES[1:]] for n in names]
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
    rows = []
    for n in names:
        d = found[n]
        p = d["params"]
        rows.append(
            [
                n,
                d["eval.clean"]["bpb"],
                p["total"],
                p["embedding"] + p["output_heads"] + p["ngram"],
                d["tokens_per_byte_train"],
                round(d["context_bytes"]),
                d["steps"],
                d["train_tokens"],
                f"{d['train_seconds'] / 60:.1f}",
            ]
        )
    return _table(head, rows)


def context_table(runs_dir: Path) -> str:
    """Corrupted-context BPB; only arms re-run after the metric was added."""
    names, found = _load(runs_dir, [e.name for e in EXPERIMENTS])
    names = [n for n in names if "ctx.clean" in found[n]]
    if not names:
        return ""
    head = [
        "experiment",
        "ctx BPB clean",
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
                f"{d['train_seconds'] / 60:.1f}",
            ]
        )
    return _table(head, rows)


def masked_table(masked_dir: Path, distill_dir: Path) -> str:
    """Runs under the active-vocabulary mask, against their unmasked twins.

    The comparison is the point of the table: the mask is justified by
    comparability across arms, not by moving any single arm's number, and the
    only way to see which of those it did is to print both.
    """
    names, found = _load(masked_dir, [t.name for t in TRANSPLANTS])
    if not names:
        return ""
    # Stage 0's native record is the unmasked twin of ``orig-frozen``: same
    # harness, same full valid split, no mask.
    ref_path = distill_dir / "native.json"
    ref = json.loads(ref_path.read_text()) if ref_path.exists() else {}
    head = [
        "experiment",
        "active / table",
        "flat floats",
        *[f"{s} masked" for s in SUITES],
        *[f"d {s} vs unmasked" for s in SUITES],
    ]
    rows = []
    for n in names:
        d = found[n]
        av = d.get("active_vocab") or {}
        twin = ref if n == "orig-frozen" else {}
        rows.append(
            [
                n,
                f"{av.get('n_active', '?')} / {av.get('table_size', '?')}",
                f"{av.get('active_flat_floats', 0) / 1e6:.1f}M",
                *[d[f"eval.{s}"]["bpb"] for s in SUITES],
                *[
                    (
                        d[f"eval.{s}"]["bpb"] - twin[f"eval.{s}"]["bpb"]
                        if twin.get(f"eval.{s}")
                        else None
                    )
                    for s in SUITES
                ],
            ]
        )
    return _table(head, rows)


def distill_table(runs_dir: Path) -> str:
    order = ["native"] + [s.name for s in DISTILLS]
    names, found = _load(runs_dir, order)
    if not names:
        return ""
    native = found.get("native", {}).get("eval.clean", {}).get("bpb")
    head = [
        "experiment",
        "method",
        "float params",
        "compress",
        "rel err",
        "rel err freq",
        "rel err active",
        "BPB clean",
        "d native",
        "d upper",
        "d title",
        "d randcase",
    ]
    rows = []
    for n in names:
        d = found[n]
        p = d["params"]
        r = d.get("recon", {})
        clean = d["eval.clean"]["bpb"]
        rows.append(
            [
                n,
                d.get("method", "-"),
                p["float_params"],
                f"x{p['compression']:.1f}",
                r.get("rel_err"),
                r.get("rel_err_freq"),
                r.get("rel_err_active"),
                clean,
                clean - native if native is not None and n != "native" else None,
                d.get("delta.upper"),
                d.get("delta.title"),
                d.get("delta.randcase"),
            ]
        )
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

    if args.distill.exists() and any(args.distill.glob("*.json")):
        parts += [
            "## 5. Stage 0 — distilling the host embedding table",
            "",
            "No LM training: each method fits the frozen Qwen3-1.7B table "
            "offline, the reconstruction replaces it (tied input + output "
            "unless noted), and the frozen body is evaluated as-is. "
            "`rel err` is the fraction of table variance the fit leaves "
            "unexplained — overall, token-frequency-weighted, and restricted "
            "to ids the eval text uses. `d native` is the BPB cost against "
            "the freshly evaluated native table. Float params only; code "
            "indices are per-token bits, reported in the JSON.",
            "",
            distill_table(args.distill),
            "",
        ]

    if args.transplant_masked.exists() and any(args.transplant_masked.glob("*.json")):
        parts += [
            "## 6. Under the active-vocabulary mask (W0 / P0)",
            "",
            "Only ids the corpus exercises stay in the stem softmax — in "
            "training and in evaluation, identically for every arm — so a "
            "per-arm BPB delta is about the arm's embedding rather than about "
            "how its rows happen to sit near the half of the host table this "
            "ru+en corpus never touches. `d ... vs unmasked` compares against "
            "the same harness with the mask lifted; masking removes terms from "
            "the normalizer, so it can only lower BPB, and the question is "
            "whether it does so by more than the 0.0016 noise floor. "
            "Compression ratios downstream are quoted against `flat floats`.",
            "",
            masked_table(args.transplant_masked, args.distill),
            "",
        ]

    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", type=Path, default=Path("results/screen2.json"))
    ap.add_argument("--runs", type=Path, default=Path("results/runs"))
    ap.add_argument("--transplant", type=Path, default=Path("results/transplant"))
    ap.add_argument(
        "--transplant-masked",
        type=Path,
        default=Path("results/transplant_masked"),
    )
    ap.add_argument("--distill", type=Path, default=Path("results/distill"))
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
