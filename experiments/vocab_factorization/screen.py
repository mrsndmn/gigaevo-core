"""Training-free screen: rank tokenizer arms in seconds instead of GPU-hours.

Measures, per (arm, perturbation suite):

``tokens_per_byte``   compression. Directly sets sequence length and therefore
                      training compute per byte of text.
``inflation``         tokens_per_byte(suite) / tokens_per_byte(clean). How
                      badly the segmentation shatters under perturbation. This
                      is the cheap proxy for downstream robustness.
``pieces_per_word``   mean stem pieces per word span.
``whole_word_rate``   fraction of word spans that survive as a single stem.
``roundtrip``         hard gate -- an arm that fails is disqualified, since a
                      lossy tokenizer's BPB is not comparable to anything.

Run::

    python -m experiments.vocab_factorization.screen --corpus data/corpus
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from experiments.vocab_factorization import perturb
from experiments.vocab_factorization.build_arms import get_arm
from experiments.vocab_factorization.configs import ARMS
from experiments.vocab_factorization.surface import KIND_WORD, segment

SCREEN_SUITES = (
    "clean",
    "typo1",
    "typo5",
    "typo10",
    "lower",
    "upper",
    "title",
    "randcase",
)


def embedding_params(arm, d_model: int, tie: bool = True) -> int:
    """Parameters spent on embeddings + output heads at a given width."""
    heads = arm.heads
    inp = heads["stem"] * d_model  # input embedding
    for name, card in heads.items():
        if name != "stem":
            inp += card * d_model  # feature embeddings, negligible but counted
    out = 0 if tie else sum(card * d_model for card in heads.values())
    return inp + out


def screen_arm(arm, texts: dict[str, str], word_sample: list[str]) -> dict:
    row: dict[str, object] = {"arm": arm.name, "vocab": arm.heads["stem"]}
    clean_tpb = None
    ok = True
    for suite, text in texts.items():
        enc = arm.encode(text)
        nbytes = len(text.encode("utf-8"))
        tpb = len(enc) / nbytes
        if suite == "clean":
            clean_tpb = tpb
        # Invertibility has to hold on perturbed text too: randcase is exactly
        # the input that exercises the CASE_OTHER verbatim fallback, and a
        # scheme that only roundtrips clean prose is not usable.
        if arm.decode(enc) != text:
            ok = False
            row[f"roundtrip_fail.{suite}"] = True
        row[f"tpb.{suite}"] = tpb
        row[f"infl.{suite}"] = tpb / clean_tpb
    row["roundtrip"] = ok

    pieces = [len(arm.encode(w)) for w in word_sample]
    row["pieces_per_word"] = sum(pieces) / len(pieces)
    row["whole_word_rate"] = sum(1 for p in pieces if p == 1) / len(pieces)
    row["heads"] = dict(arm.heads)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    ap.add_argument("--arms-dir", type=Path, default=Path("data/arms"))
    ap.add_argument("--split", default="valid")
    ap.add_argument("--eval-mb", type=float, default=2.0)
    ap.add_argument("--word-sample", type=int, default=30000)
    ap.add_argument("--out", type=Path, default=Path("results/screen.json"))
    args = ap.parse_args()

    from experiments.vocab_factorization.data import load_split

    text = load_split(args.corpus, args.split)
    text = text[: int(args.eval_mb * 1024 * 1024)]
    text = text[: text.rfind("\n") + 1]
    print(f"screen text: {len(text.encode('utf-8')) / 1e6:.2f} MB", flush=True)

    texts = {s: perturb.apply_suite(s, text, seed=0) for s in SCREEN_SUITES}
    words = [sp.text for sp in segment(text) if sp.kind == KIND_WORD][
        : args.word_sample
    ]

    rows = []
    for spec in ARMS:
        t = time.time()
        arm = get_arm(spec, args.arms_dir, args.corpus)
        row = screen_arm(arm, texts, words)
        row["seconds"] = round(time.time() - t, 1)
        row["embed_params_d512_tied"] = embedding_params(arm, 512, tie=True)
        rows.append(row)
        print(
            f"{spec.name:16s} tpb={row['tpb.clean']:.4f} "
            f"ppw={row['pieces_per_word']:.3f} "
            f"upper_infl={row['infl.upper']:.3f} typo5_infl={row['infl.typo5']:.3f} "
            f"roundtrip={row['roundtrip']}",
            flush=True,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
