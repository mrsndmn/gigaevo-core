"""Build and cache tokenizer arms from ``configs.ARMS``.

Arms are cached on disk by name so the screen, the data prep, and every
training run share byte-identical tokenizers.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from experiments.vocab_factorization.configs import (
    ARMS,
    ARMS_BY_NAME,
    TOKENIZER_TRAIN_BYTES,
    ArmSpec,
)
from experiments.vocab_factorization.tokenization import (
    BPEArm,
    ByteArm,
    FactorizedArm,
    load_arm,
)

_CACHE: dict[str, object] = {}


def _train_sample(corpus_dir: Path, nbytes: int) -> str:
    """Equal-sized ru and en halves, so neither language dominates the merges."""

    half = nbytes // 2
    parts = []
    for lang in ("ru", "en"):
        p = corpus_dir / f"train.{lang}.txt"
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as f:
            s = f.read(half)
        parts.append(s[: s.rfind("\n") + 1])
    if not parts:
        raise FileNotFoundError(f"no training shards under {corpus_dir}")
    return "".join(parts)


def build_arm(spec: ArmSpec, corpus: str):
    if spec.kind == "byte":
        return ByteArm()
    if spec.kind == "bpe":
        return BPEArm.train(corpus, spec.vocab_size, name=spec.name)
    if spec.kind == "factorized":
        return FactorizedArm.train(
            corpus,
            spec.vocab_size,
            n_suffix=spec.n_suffix,
            use_suffix=spec.use_suffix,
            piece_case=spec.piece_case,
            name=spec.name,
        )
    raise ValueError(f"unknown arm kind {spec.kind!r}")


def get_arm(spec: ArmSpec | str, arms_dir: Path, corpus_dir: Path):
    """Load a cached arm, training and saving it on first use.

    A name that is not in ``ARMS`` but exists on disk is loaded directly: that
    is how externally-built arms (a host model's native tokenizer, saved as an
    HFArm) enter the pipeline without needing a build recipe here.
    """
    if isinstance(spec, str):
        if spec not in ARMS_BY_NAME:
            path = arms_dir / spec
            if (path / "arm.json").exists():
                if spec not in _CACHE:
                    _CACHE[spec] = load_arm(path)
                return _CACHE[spec]
            raise KeyError(f"unknown arm {spec!r} and nothing cached at {path}")
        spec = ARMS_BY_NAME[spec]
    if spec.name in _CACHE:
        return _CACHE[spec.name]
    path = arms_dir / spec.name
    if (path / "arm.json").exists():
        arm = load_arm(path)
    else:
        sample = _train_sample(corpus_dir, TOKENIZER_TRAIN_BYTES)
        arm = build_arm(spec, sample)
        arm.save(path)
    _CACHE[spec.name] = arm
    return arm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    ap.add_argument("--out", type=Path, default=Path("data/arms"))
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    specs = ARMS if not args.only else [ARMS_BY_NAME[n] for n in args.only]
    sample = None
    for spec in specs:
        path = args.out / spec.name
        if (path / "arm.json").exists():
            print(f"{spec.name:16s} cached")
            continue
        if sample is None:
            sample = _train_sample(args.corpus, TOKENIZER_TRAIN_BYTES)
            print(f"tokenizer training sample: {len(sample.encode()) / 1e6:.1f} MB")
        t = time.time()
        arm = build_arm(spec, sample)
        arm.save(path)
        print(f"{spec.name:16s} built in {time.time() - t:5.1f}s  heads={arm.heads}")


if __name__ == "__main__":
    main()
