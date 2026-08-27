"""Figure: vocabulary transplant onto a frozen Qwen3-1.7B.

Panel A carries two quantities per arm — bits-per-byte before any training and
after — so the "swap the vocabulary on the fly, train nothing" result is
visible next to the trained one. Step 0 is drawn as an outlined marker rather
than a second colour, because colour already encodes arm identity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#dedcd5"

ARMS = ["orig-frozen", "orig-reinit", "bpe8k-t-random", "bpe8k-t", "fact8k-nosuf-t"]
LABELS = {
    "orig-frozen": "Qwen3 native, untouched",
    "orig-reinit": "Qwen3 native, re-init (control)",
    "bpe8k-t-random": "BPE 8k, random init",
    "bpe8k-t": "BPE 8k, mean-of-pieces",
    "fact8k-nosuf-t": "Factorized 8k, mean-of-pieces",
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("results/transplant"))
    ap.add_argument("--out", type=Path, default=Path("results/vocab_transplant.png"))
    args = ap.parse_args()

    data = {}
    for p in args.runs.glob("*.json"):
        d = json.loads(p.read_text())
        data[d["experiment"]] = d
    arms = [a for a in ARMS if a in data]
    if not arms:
        raise SystemExit(f"no results in {args.runs}")

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(16.2, 6.4), gridspec_kw={"width_ratios": [1.05, 2.0]}
    )
    fig.patch.set_facecolor(SURFACE)

    # ---- Panel A: clean BPB, before and after ------------------------------
    ax1.set_facecolor(SURFACE)
    final = [data[a]["eval.clean"]["bpb"] for a in arms]
    pre = [data[a]["step0_clean_bpb"] for a in arms]
    y = list(range(len(arms)))
    bars = ax1.barh(y, final, height=0.6, color=[SERIES[ARMS.index(a)] for a in arms])
    for b in bars:
        b.set_linewidth(1.5)
        b.set_edgecolor(SURFACE)
    ax1.scatter(
        pre,
        y,
        marker="D",
        s=46,
        facecolors="none",
        edgecolors=INK,
        linewidths=1.4,
        zorder=5,
        label="before training (step 0)",
    )
    for i, (v, p) in enumerate(zip(final, pre, strict=True)):
        # Nudge past the step-0 diamond when the two nearly coincide.
        off = 0.20 if abs(p - v) < 0.12 else 0.09
        ax1.text(v + off, i, f"{v:.3f}", va="center", color=INK, fontsize=10.5)
    ref = data["orig-frozen"]["eval.clean"]["bpb"] if "orig-frozen" in data else None
    if ref:
        ax1.axvline(ref, color=INK2, linewidth=1.1, linestyle=(0, (4, 3)), zorder=1)
        ax1.text(
            ref + 0.07,
            -0.62,
            "pretrained reference",
            fontsize=8.8,
            color=INK2,
            va="center",
        )
    ax1.set_yticks(y)
    ax1.set_yticklabels([LABELS[a] for a in arms], fontsize=10, color=INK)
    ax1.set_ylim(len(arms) - 0.4, -0.95)  # headroom for the reference label
    ax1.set_xlabel("bits per byte (lower is better)", fontsize=10.5, color=INK2)
    ax1.set_title(
        "A · Clean text", fontsize=13, color=INK, loc="left", pad=12, fontweight="bold"
    )
    ax1.set_xlim(0, max(max(pre), max(final)) * 1.12)
    ax1.grid(axis="x", color=GRID, linewidth=0.8)
    ax1.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax1.spines[s].set_visible(False)
    ax1.spines["bottom"].set_color(GRID)
    ax1.tick_params(colors=INK2, length=0)
    ax1.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK)

    # ---- Panel B: delta BPB by suite ---------------------------------------
    ax2.set_facecolor(SURFACE)
    n = len(arms)
    width = 0.8 / n
    for i, arm in enumerate(arms):
        xs = [j + (i - (n - 1) / 2) * width for j in range(len(SUITES))]
        vs = [data[arm][f"delta.{s}"] for s in SUITES]
        bb = ax2.bar(
            xs, vs, width=width * 0.9, color=SERIES[ARMS.index(arm)], label=LABELS[arm]
        )
        for b in bb:
            b.set_linewidth(1.0)
            b.set_edgecolor(SURFACE)
        for x, v in zip(xs, vs, strict=True):
            ax2.text(
                x,
                v + 0.03,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.2,
                color=INK2,
                rotation=90,
            )
    ax2.set_xticks(range(len(SUITES)))
    ax2.set_xticklabels([SUITE_LABELS[s] for s in SUITES], fontsize=10.5, color=INK)
    ax2.set_ylabel("Δ bits per byte vs clean", fontsize=10.5, color=INK2)
    ax2.set_title(
        "B · Cost of corruption (lower is more robust)",
        fontsize=13,
        color=INK,
        loc="left",
        pad=12,
        fontweight="bold",
    )
    ax2.grid(axis="y", color=GRID, linewidth=0.8)
    ax2.set_axisbelow(True)
    ax2.axhline(0, color=GRID, linewidth=1.2)
    for s in ("top", "right", "left"):
        ax2.spines[s].set_visible(False)
    ax2.spines["bottom"].set_color(GRID)
    ax2.tick_params(colors=INK2, length=0)
    ax2.legend(frameon=False, fontsize=9.5, ncol=2, loc="upper left", labelcolor=INK)
    top = max(data[a][f"delta.{s}"] for a in arms for s in SUITES)
    ax2.set_ylim(0, top * 1.34)

    fig.suptitle(
        "Vocabulary transplant onto a frozen Qwen3-1.7B: 19x smaller embedding, "
        "no loss in bits-per-byte",
        fontsize=14.5,
        color=INK,
        x=0.011,
        ha="left",
        y=0.985,
        fontweight="bold",
    )
    fig.text(
        0.011,
        0.925,
        "Transformer body frozen; only the embedding table and heads train, on "
        "241 MB of ru+en Wikipedia (~56M tokens, ~51 min on one A100).",
        fontsize=10,
        color=INK2,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170, facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
