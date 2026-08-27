"""Figure: clean bits-per-byte, and the robustness cost of each perturbation.

Two panels because the two quantities answer different questions and share no
scale: panel A is "what does this representation cost on clean text", panel B
is "what does it cost when the text is corrupted". Plotting them on one axis
would be a dual-axis chart.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Validated categorical slots (light mode) from the dataviz reference palette.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#dedcd5"

# The five arms that carry the argument. bpe8k / fact8k / the n-gram runs are
# in the tables; putting nine series on one axis would be unreadable.
ARMS = ["bpe32k", "bpe8k", "fact8k-nosuf", "fact16k", "byte"]
LABELS = {
    "bpe32k": "BPE 32k — reference",
    "bpe8k": "BPE 8k — null hypothesis",
    "fact8k-nosuf": "Factorized 8k — case+space",
    "fact16k": "Factorized 16k — +suffix head",
    "byte": "Bytes",
}
SUITES = ["typo1", "typo5", "lower", "title", "upper", "randcase"]
SUITE_LABELS = {
    "typo1": "typo 1%",
    "typo5": "typo 5%",
    "lower": "all-lower",
    "title": "Title Case",
    "upper": "ALL-UPPER",
    "randcase": "rAnDoM case",
}


def load(runs: Path) -> dict:
    out = {}
    for p in runs.glob("*.json"):
        d = json.loads(p.read_text())
        out[d["experiment"]] = d
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("results/runs"))
    ap.add_argument("--out", type=Path, default=Path("results/vocab_factorization.png"))
    args = ap.parse_args()

    data = load(args.runs)
    arms = [a for a in ARMS if a in data]
    if not arms:
        raise SystemExit(f"no results in {args.runs}")

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15.5, 6.2), gridspec_kw={"width_ratios": [1, 2.05]}
    )
    fig.patch.set_facecolor(SURFACE)

    # ---- Panel A: clean BPB ------------------------------------------------
    ax1.set_facecolor(SURFACE)
    vals = [data[a]["eval.clean"]["bpb"] for a in arms]
    y = range(len(arms))
    bars = ax1.barh(
        list(y), vals, height=0.62, color=[SERIES[ARMS.index(a)] for a in arms]
    )
    for b in bars:
        b.set_linewidth(1.5)
        b.set_edgecolor(SURFACE)
    for i, v in enumerate(vals):
        ax1.text(v + max(vals) * 0.015, i, f"{v:.3f}", va="center", color=INK, fontsize=11)
    ax1.set_yticks(list(y))
    ax1.set_yticklabels([LABELS[a] for a in arms], fontsize=10.5, color=INK)
    ax1.invert_yaxis()
    ax1.set_xlabel("bits per byte (lower is better)", fontsize=10.5, color=INK2)
    ax1.set_title(
        "A · Clean text", fontsize=13, color=INK, loc="left", pad=12, fontweight="bold"
    )
    ax1.set_xlim(0, max(vals) * 1.18)
    ax1.grid(axis="x", color=GRID, linewidth=0.8)
    ax1.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax1.spines[s].set_visible(False)
    ax1.spines["bottom"].set_color(GRID)
    ax1.tick_params(colors=INK2, length=0)

    # ---- Panel B: delta BPB by suite ---------------------------------------
    ax2.set_facecolor(SURFACE)
    n = len(arms)
    width = 0.78 / n
    for i, arm in enumerate(arms):
        xs = [j + (i - (n - 1) / 2) * width for j in range(len(SUITES))]
        vs = [data[arm][f"delta.{s}"] for s in SUITES]
        bb = ax2.bar(
            xs, vs, width=width * 0.9, color=SERIES[ARMS.index(arm)], label=LABELS[arm]
        )
        for b in bb:
            b.set_linewidth(1.0)
            b.set_edgecolor(SURFACE)
        # Direct labels are the relief for the contrast WARN on slots 3-5.
        for x, v in zip(xs, vs, strict=True):
            ax2.text(
                x, v + 0.035, f"{v:.2f}", ha="center", va="bottom",
                fontsize=7.4, color=INK2, rotation=90,
            )
    ax2.set_xticks(range(len(SUITES)))
    ax2.set_xticklabels([SUITE_LABELS[s] for s in SUITES], fontsize=10.5, color=INK)
    ax2.set_ylabel("Δ bits per byte vs clean", fontsize=10.5, color=INK2)
    ax2.set_title(
        "B · Cost of corruption (lower is more robust)",
        fontsize=13, color=INK, loc="left", pad=12, fontweight="bold",
    )
    ax2.grid(axis="y", color=GRID, linewidth=0.8)
    ax2.set_axisbelow(True)
    ax2.axhline(0, color=GRID, linewidth=1.2)
    for s in ("top", "right", "left"):
        ax2.spines[s].set_visible(False)
    ax2.spines["bottom"].set_color(GRID)
    ax2.tick_params(colors=INK2, length=0)
    ax2.legend(
        frameon=False, fontsize=10, ncol=2, loc="upper left", labelcolor=INK
    )
    top = max(
        data[a][f"delta.{s}"] for a in arms for s in SUITES
    )
    ax2.set_ylim(min(0, min(data[a][f"delta.{s}"] for a in arms for s in SUITES)) - 0.05, top * 1.30)

    fig.suptitle(
        "Vocabulary factorization: case robustness is nearly free, morphology is not",
        fontsize=14.5, color=INK, x=0.011, ha="left", y=0.985, fontweight="bold",
    )
    fig.text(
        0.011, 0.925,
        "33M-param LM, ru+en Wikipedia, identical 400 MB x 3 epochs per arm. "
        "Bits per byte sums every prediction head.",
        fontsize=10, color=INK2, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170, facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
