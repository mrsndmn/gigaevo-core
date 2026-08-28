"""Encode the corpus once per arm and cache the token arrays.

Encoding is pure Python per span, so a 150 MB pass is minutes single-core;
this module fans it out over a process pool. Chunks are cut on newlines, so
the only difference from encoding the whole string at once is at chunk
boundaries -- one per 4 MB, and identical for every arm.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

from experiments.vocab_factorization.configs import (
    ARMS,
    ARMS_BY_NAME,
    token_cache_path,
)
from experiments.vocab_factorization.tokenization import Encoding, iter_chunks

__all__ = [
    "build_text",
    "encode_text",
    "get_tokens",
    "token_cache_path",
]

_WORKER_ARM = None


def _init_worker(arm_name: str, arms_dir: str) -> None:
    global _WORKER_ARM
    from experiments.vocab_factorization.tokenization import load_arm

    _WORKER_ARM = load_arm(Path(arms_dir) / arm_name)


def _encode_chunk(chunk: str):
    enc = _WORKER_ARM.encode(chunk)
    return enc.stem, enc.role, enc.suffix


def build_text(
    corpus_dir: Path, split: str, nbytes: int | None, block: int = 1 << 18
) -> str:
    """Read a split as ru/en interleaved in fixed blocks.

    Interleaving matters: random windows sampled from a ru-then-en
    concatenation would be single-language, and the model would never see a
    batch that mixes scripts.
    """
    parts = {}
    for lang in ("ru", "en"):
        p = corpus_dir / f"{split}.{lang}.txt"
        if p.exists():
            parts[lang] = p.read_text(encoding="utf-8")
    if not parts:
        raise FileNotFoundError(f"no shards for split {split!r} in {corpus_dir}")

    out: list[str] = []
    total = 0
    pos = dict.fromkeys(parts, 0)
    done = False
    while not done:
        done = True
        for lang, txt in parts.items():
            i = pos[lang]
            if i >= len(txt):
                continue
            j = txt.find("\n", i + block)
            j = len(txt) if j == -1 else j + 1
            piece = txt[i:j]
            out.append(piece)
            pos[lang] = j
            total += len(piece.encode("utf-8"))
            done = False
            if nbytes is not None and total >= nbytes:
                return "".join(out)
    return "".join(out)


def encode_text(
    arm_name: str, arms_dir: Path, text: str, workers: int = 32
) -> Encoding:
    import multiprocessing as mp

    chunks = list(iter_chunks(text, 1 << 22))
    if len(chunks) <= 1:
        # Evaluation texts are a few MB and get encoded once per suite; a pool
        # spawn costs far more than the encode itself.
        _init_worker(arm_name, str(arms_dir))
        stem, role, suffix = _encode_chunk(chunks[0] if chunks else "")
        return Encoding(stem=stem, role=role, suffix=suffix)
    ctx = mp.get_context("fork")
    with ctx.Pool(
        processes=min(workers, len(chunks)),
        initializer=_init_worker,
        initargs=(arm_name, str(arms_dir)),
    ) as pool:
        results = pool.map(_encode_chunk, chunks, chunksize=1)
    return Encoding(
        stem=np.concatenate([r[0] for r in results]),
        role=np.concatenate([r[1] for r in results]),
        suffix=np.concatenate([r[2] for r in results]),
    )


def get_tokens(
    arm_name: str,
    split: str,
    nbytes: int | None,
    corpus_dir: Path,
    arms_dir: Path,
    tokens_dir: Path,
    workers: int = 32,
) -> tuple[Encoding, int]:
    """Return (encoding, byte count of the encoded text), caching to disk."""
    path = token_cache_path(tokens_dir, arm_name, split, nbytes)
    if path.exists():
        z = np.load(path)
        return Encoding(stem=z["stem"], role=z["role"], suffix=z["suffix"]), int(
            z["nbytes"]
        )
    text = build_text(corpus_dir, split, nbytes)
    n = len(text.encode("utf-8"))
    enc = encode_text(arm_name, arms_dir, text, workers=workers)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, stem=enc.stem, role=enc.role, suffix=enc.suffix, nbytes=n)
    return enc, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    ap.add_argument("--arms-dir", type=Path, default=Path("data/arms"))
    ap.add_argument("--tokens-dir", type=Path, default=Path("data/tokens"))
    ap.add_argument("--train-mb", type=int, default=150)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    specs = ARMS if not args.only else [ARMS_BY_NAME[n] for n in args.only]
    nbytes = args.train_mb * 1024 * 1024
    for spec in specs:
        for split, budget in (("train", nbytes), ("valid", None)):
            t = time.time()
            enc, nb = get_tokens(
                spec.name,
                split,
                budget,
                args.corpus,
                args.arms_dir,
                args.tokens_dir,
                workers=args.workers,
            )
            print(
                f"{spec.name:16s} {split:5s} {len(enc) / 1e6:8.2f}M tokens "
                f"{nb / 1e6:7.2f} MB  tpb={len(enc) / nb:.4f}  ({time.time() - t:.1f}s)",
                flush=True,
            )


if __name__ == "__main__":
    main()
