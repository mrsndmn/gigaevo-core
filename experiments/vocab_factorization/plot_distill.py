"""Figure for stage 0: BPB damage vs float-parameter compression.

One line per method family, log-log, with the pre-registered kill criterion
drawn where it was set (0.05 BPB at ~x10). The renormed controls sit far above
everything: restoring native row norms amplifies wrong directions, which is
what shows the damage is directional rather than a softmax-temperature
artifact.

    python -m experiments.vocab_factorization.plot_distill
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
CRITICAL = "#d03b3b"

FAMILIES = {
    # label -> (hue, member experiments in compression order)
    "SVD (global low-rank)": (
        "#2a78d6",
        ["svd512", "svd256", "svd128", "svd64"],
    ),
    "RVQ (additive codes)": (
        "#eb6834",
        ["rvq8x4k", "rvq4x4k", "rvq2x4k", "vq4k"],
    ),
    "tree + low-rank residual": (
        "#1baf7a",
        ["tree64x2-r448", "tree64x2-r192", "tree64x2-r64", "tree64x2-r8", "tree64x2"],
    ),
}
FREQ = {"svd128-freq": "#2a78d6", "tree64x2-r64-freq": "#1baf7a"}
RENORM = ["svd128-freq-norm", "tree64x2-r64-freq-norm"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results/distill"))
    ap.add_argument("--out", type=Path, default=Path("results/distill_stage0.png"))
    args = ap.parse_args()

    found = {}
    for p in args.results.glob("*.json"):
        d = json.loads(p.read_text())
        found[d["experiment"]] = d

    def xy(name: str) -> tuple[float, float] | None:
        d = found.get(name)
        if not d or "dnative.clean" not in d:
            return None
        return d["params"]["compression"], d["dnative.clean"]

    fig, ax = plt.subplots(figsize=(9.0, 5.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for label, (color, names) in FAMILIES.items():
        pts = [p for p in (xy(n) for n in names) if p]
        pts.sort()
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=color, lw=2, marker="o", ms=6, label=label, zorder=3)

    freq_offsets = {"svd128-freq": (0, 10), "tree64x2-r64-freq": (0, -16)}
    for name, color in FREQ.items():
        p = xy(name)
        if p:
            ax.plot(*p, marker="o", ms=8, mfc=SURFACE, mec=color, mew=2, zorder=4)
            ax.annotate(
                "freq-weighted fit",
                p,
                xytext=freq_offsets[name],
                textcoords="offset points",
                color=color,
                fontsize=8,
                ha="center",
            )

    ren = [p for p in (xy(n) for n in RENORM) if p]
    if ren:
        xs, ys = zip(*ren)
        ax.plot(
            xs,
            ys,
            ls="none",
            marker="x",
            ms=7,
            color=MUTED,
            mew=2,
            zorder=3,
            label="renormed to native norms (control)",
        )
        ax.annotate(
            "renormed controls: wrong\ndirections at full scale",
            (xs[-1], ys[-1]),
            xytext=(12, -4),
            textcoords="offset points",
            color=MUTED,
            fontsize=8,
            ha="left",
            va="top",
        )

    ax.axhline(0.05, color=CRITICAL, lw=1.2, ls=(0, (4, 3)), zorder=2)
    ax.annotate(
        "kill criterion: 0.05 BPB at ~x10 — every fit is 30-60x above it",
        (2.6, 0.05),
        xytext=(0, 5),
        textcoords="offset points",
        color=CRITICAL,
        fontsize=8.5,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(2.5, 260)
    ax.set_ylim(0.03, 11)
    ax.set_xticks([4, 8, 16, 32, 64, 128])
    ax.set_xticklabels(["x4", "x8", "x16", "x32", "x64", "x128"])
    ax.set_yticks([0.05, 0.1, 0.3, 1, 3, 10])
    ax.set_yticklabels(["0.05", "0.1", "0.3", "1", "3", "10"])
    ax.tick_params(colors=MUTED, labelsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASELINE)
    ax.grid(True, which="major", color=GRID, lw=0.7)
    ax.set_axisbelow(True)

    ax.set_xlabel("float-parameter compression of the embedding table", color=INK)
    ax.set_ylabel("clean-BPB damage vs native (frozen body)", color=INK)
    ax.set_title(
        "Stage 0: offline distillation of Qwen3-1.7B's embedding table fails "
        "at every compression",
        color=INK,
        fontsize=11,
        loc="left",
        pad=12,
    )
    ax.text(
        0.985,
        0.03,
        "reference: a 16.4M-float table (x19) trained through the frozen body\n"
        "beats native BPB (study 2, bpe8k-t) — capacity is not the obstacle",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=8,
        ha="right",
        va="bottom",
    )
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
