"""Methods figure: how the factorized tokenizer encodes, and how BPB is scored.

Every string shown comes from ``results/example.json``, which is produced by
``dump_example.py`` running the real tokenizers, and every number in panel C
comes from ``results/runs/*.json``. The figure cannot drift from the code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.patches import FancyBboxPatch  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#dedcd5"
FAINT = "#f0efea"
MONO = "DejaVu Sans Mono"


def box(ax, x, y, w, h, fc, ec=None, lw=1.0, r=0.012, alpha=1.0):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        linewidth=lw,
        facecolor=fc,
        edgecolor=ec or fc,
        alpha=alpha,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(p)
    return p


def arrow(ax, x, y0, y1, label=""):
    ax.annotate(
        "",
        xy=(x, y1),
        xytext=(x, y0),
        xycoords="axes fraction",
        arrowprops={
            "arrowstyle": "-|>",
            "color": INK2,
            "linewidth": 1.3,
            "shrinkA": 0,
            "shrinkB": 0,
        },
    )
    if label:
        ax.text(
            x + 0.022,
            (y0 + y1) / 2,
            label,
            transform=ax.transAxes,
            fontsize=8.6,
            color=INK2,
            va="center",
            style="italic",
        )


def clean_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_facecolor(SURFACE)


def panel_pipeline(ax, ex):
    clean_axes(ax)
    ax.set_title(
        "A · Encoding one word, losslessly",
        fontsize=12.5,
        color=INK,
        loc="left",
        pad=10,
        fontweight="bold",
    )

    steps = [
        ("input text", "В городах России живут миллионы людей.", FAINT, INK),
        (
            "segment()  — regex spans that concatenate back exactly",
            "'В'  ' городах'  ' России'  …  '.'",
            FAINT,
            INK,
        ),
        (
            "pull out features, casefold the core",
            "' городах'  →  space=1   case=lower   core='городах'",
            "#e3f3ec",
            INK,
        ),
        (
            "verify restore(case, core) == original\n"
            "else classify OTHER and keep the bytes verbatim",
            "'iPhone', 'STRAßE'  →  verbatim path",
            "#fdf0e8",
            INK,
        ),
        (
            "split stem + suffix, only if the stem is one BPE piece",
            "'городах'  →  stem 'город'  +  suffix 'ах'",
            "#fdf3dc",
            INK,
        ),
        ("one token row", "stem='город'   role='␣lower'   suffix='ах'", "#dceaf9", INK),
    ]
    y = 0.955
    for i, (cap, val, fc, tc) in enumerate(steps):
        nlines = cap.count("\n") + 1
        h = 0.096 + 0.028 * (nlines - 1)
        box(ax, 0.02, y - h, 0.96, h, fc)
        ax.text(
            0.045,
            y - 0.026 - 0.028 * (nlines - 1) / 2,
            cap,
            transform=ax.transAxes,
            fontsize=8.7,
            color=INK2,
            va="center",
        )
        ax.text(
            0.045,
            y - h + 0.030,
            val,
            transform=ax.transAxes,
            fontsize=9.4,
            color=tc,
            va="center",
            family=MONO,
        )
        if i < len(steps) - 1:
            arrow(ax, 0.075, y - h, y - h - 0.040)
        y -= h + 0.040

    box(ax, 0.02, 0.012, 0.96, 0.058, "#eef6f2", ec=AQUA, lw=1.2)
    ax.text(
        0.5,
        0.041,
        "decode(encode(x)) == x, byte-exact — enforced on clean AND perturbed text",
        transform=ax.transAxes,
        fontsize=9.2,
        color=INK,
        ha="center",
        va="center",
        fontweight="bold",
    )


def _pieces(rows, maxn=None):
    out = []
    for r in rows:
        p = r["piece"]
        p = "·" if ("�" in p or not p.strip()) and p.strip() == "" else p
        p = p.replace("�", "·")
        out.append((p, r.get("role", "")))
    if maxn and len(out) > maxn:
        return out[:maxn], len(rows)
    return out, len(rows)


def panel_streams(ax, ex):
    clean_axes(ax)
    ax.set_title(
        "B · What uppercasing does to the token stream",
        fontsize=12.5,
        color=INK,
        loc="left",
        pad=10,
        fontweight="bold",
    )

    y = 0.90
    for label in ("clean", "upper"):
        st = ex["streams"][label]
        ax.text(
            0.005,
            y + 0.055,
            st["text"],
            transform=ax.transAxes,
            fontsize=9.6,
            color=INK,
            family=MONO,
        )
        for name, color in (("bpe16k", BLUE), ("fact16k", AQUA)):
            rows, total = _pieces(st[name])
            n = len(rows)
            gap = 0.006
            # 0.74, not 0.80: the token-count label lives at x=0.985 and the
            # 32-box uppercase row would otherwise run underneath it.
            w = min(0.085, (0.74 - gap * (n - 1)) / max(n, 1))
            x = 0.155
            for p, role in rows:
                fc = color if name == "bpe16k" else AQUA
                box(ax, x, y - 0.052, w, 0.048, fc, alpha=0.20)
                if w > 0.02:
                    ax.text(
                        x + w / 2,
                        y - 0.028,
                        p,
                        transform=ax.transAxes,
                        fontsize=7.8,
                        color=INK,
                        ha="center",
                        va="center",
                        family=MONO,
                    )
                if name == "fact16k":
                    rc = (
                        ORANGE
                        if "upper" in role
                        else (YELLOW if "title" in role else GRID)
                    )
                    box(ax, x, y - 0.070, w, 0.013, rc)
                x += w + gap
            ax.text(
                0.005,
                y - 0.028,
                name,
                transform=ax.transAxes,
                fontsize=9.2,
                color=INK,
                va="center",
            )
            ax.text(
                0.985,
                y - 0.028,
                f"{total} tok",
                transform=ax.transAxes,
                fontsize=9.6,
                color=INK,
                va="center",
                ha="right",
                fontweight="bold",
            )
            y -= 0.135
        y -= 0.075

    ax.text(
        0.005,
        0.075,
        "The factorized stem row is byte-identical between the two blocks. "
        "Only the role strip changes\n"
        "(grey = lower, gold = Title, orange = UPPER). BPE has no such "
        "field, so every word shatters into letters.",
        transform=ax.transAxes,
        fontsize=9.0,
        color=INK2,
        va="center",
    )
    ax.text(
        0.005,
        0.005,
        "· = a byte-level BPE piece that is not a whole character",
        transform=ax.transAxes,
        fontsize=7.8,
        color=INK2,
        va="center",
    )


def panel_bpb(ax, runs):
    clean_axes(ax)
    ax.set_title(
        "C · Bits per byte sums every prediction head",
        fontsize=12.5,
        color=INK,
        loc="left",
        pad=10,
        fontweight="bold",
    )
    ax.set_xticks([])

    order = ["bpe16k", "fact8k-nosuf", "fact16k"]
    head_color = {"stem": BLUE, "role": YELLOW, "suffix": ORANGE}
    # Colour key: the role head is too thin to carry an inline label.
    kx = 0.20
    for h, c in head_color.items():
        box(ax, kx, 0.855, 0.022, 0.030, c)
        ax.text(
            kx + 0.030,
            0.870,
            f"{h} head",
            transform=ax.transAxes,
            fontsize=8.8,
            color=INK2,
            va="center",
        )
        kx += 0.135
    xmax = 1.55
    y = 0.78
    for name in order:
        e = runs[name]["eval.clean"]
        heads = {k.split(".")[1]: v for k, v in e.items() if k.startswith("bpb.")}
        x = 0.20
        for h, v in heads.items():
            w = v / xmax * 0.66
            box(ax, x, y - 0.085, w, 0.085, head_color[h])
            if w > 0.05:
                ax.text(
                    x + w / 2,
                    y - 0.0425,
                    f"{h}\n{v:.3f}",
                    transform=ax.transAxes,
                    fontsize=8.3,
                    color="white",
                    ha="center",
                    va="center",
                    fontweight="bold",
                )
            x += w + 0.004
        ax.text(
            0.19,
            y - 0.0425,
            name,
            transform=ax.transAxes,
            fontsize=9.6,
            color=INK,
            va="center",
            ha="right",
        )
        ax.text(
            x + 0.012,
            y - 0.0425,
            f"= {e['bpb']:.4f}",
            transform=ax.transAxes,
            fontsize=10,
            color=INK,
            va="center",
            fontweight="bold",
        )
        y -= 0.145

    ax.text(
        0.02,
        0.20,
        "Moving case or morphology into a side head is never free: the head "
        "still has to predict it,\nand its NLL is inside the same sum. "
        "That is what makes the arms comparable at all.",
        transform=ax.transAxes,
        fontsize=9.0,
        color=INK2,
        va="center",
    )
    ax.text(
        0.02,
        0.045,
        "Two-stage evaluation:  training-free screen (seconds/arm, "
        "compression + fragmentation + the lossless gate)\n"
        "→  train a 33M LM per surviving arm (~15 min on one A100).",
        transform=ax.transAxes,
        fontsize=9.0,
        color=INK,
        va="center",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--example", type=Path, default=Path("results/example.json"))
    ap.add_argument("--runs", type=Path, default=Path("results/runs"))
    ap.add_argument(
        "--out", type=Path, default=Path("results/vocab_factorization_method.png")
    )
    args = ap.parse_args()

    ex = json.loads(args.example.read_text())
    runs = {}
    for p in args.runs.glob("*.json"):
        d = json.loads(p.read_text())
        runs[d["experiment"]] = d

    fig = plt.figure(figsize=(16.5, 9.6))
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.0, 1.32],
        height_ratios=[1.30, 1.0],
        left=0.012,
        right=0.988,
        top=0.885,
        bottom=0.02,
        wspace=0.06,
        hspace=0.16,
    )
    panel_pipeline(fig.add_subplot(gs[:, 0]), ex)
    panel_streams(fig.add_subplot(gs[0, 1]), ex)
    panel_bpb(fig.add_subplot(gs[1, 1]), runs)

    fig.suptitle(
        "Method: factorize the surface form, then make every factor pay for itself",
        fontsize=15,
        color=INK,
        x=0.012,
        ha="left",
        y=0.965,
        fontweight="bold",
    )
    fig.text(
        0.012,
        0.918,
        "Stem vocabulary + an 8-value role head (leading space × case) "
        "+ an optional suffix head. ru+en Wikipedia, 33M-param LM.",
        fontsize=10.2,
        color=INK2,
        ha="left",
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=165, facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
