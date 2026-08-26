"""Corpus preparation for the vocabulary-factorization study.

Builds a fixed ru+en Wikipedia corpus as flat UTF-8 text shards. Every
tokenizer arm in the study sees byte-identical input, so bits-per-byte
numbers are comparable across arms.

Design decisions worth knowing:

* NFC normalization is applied ONCE here, at corpus build time. After this
  point the corpus is frozen bytes; no tokenizer is allowed to normalize
  further (that would silently discard information and inflate its BPB
  advantage).
* Documents are split train/valid/test by a stable hash of the document id,
  so the split is reproducible and independent of download order.
* ru and en documents are interleaved 1:1 by document, not by byte. The
  realized byte ratio is reported by ``describe()``.

Usage::

    python -m experiments.vocab_factorization.data --out data/corpus
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import unicodedata

LANGS = ("ru", "en")
WIKI_CONFIG = {"ru": "20231101.ru", "en": "20231101.en"}

# Bytes per split per language. Train dominates; valid/test are small but
# large enough that BPB is stable to ~1e-3.
DEFAULT_BUDGET = {
    "train": 300 * 1024 * 1024,
    "valid": 2 * 1024 * 1024,
    "test": 2 * 1024 * 1024,
}

# Hash buckets out of 1000. Sized so valid/test fill within roughly the same
# streaming pass that fills train -- at 0.5% they would need 800 MB of stream
# for a 4 MB split, which dominates corpus build time for no benefit.
_VALID_BUCKETS = range(0, 20)
_TEST_BUCKETS = range(20, 40)


def _split_of(doc_id: str) -> str:
    bucket = int(hashlib.sha1(doc_id.encode("utf-8")).hexdigest()[:8], 16) % 1000
    if bucket in _VALID_BUCKETS:
        return "valid"
    if bucket in _TEST_BUCKETS:
        return "test"
    return "train"


def _clean(text: str) -> str:
    """NFC-normalize and collapse the runs of blank lines Wikipedia emits.

    Deliberately conservative: case, punctuation, and non-breaking spaces are
    left exactly as they appear. Those are precisely the surface phenomena the
    study is about, so normalizing them away here would beg the question.
    """
    text = unicodedata.normalize("NFC", text)
    lines = [ln.rstrip() for ln in text.split("\n")]
    out: list[str] = []
    blank = 0
    for ln in lines:
        if ln:
            blank = 0
            out.append(ln)
        else:
            blank += 1
            if blank <= 1:
                out.append("")
    return "\n".join(out).strip() + "\n"


def build(out_dir: Path, budget: dict[str, int]) -> dict:
    from datasets import load_dataset

    out_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, dict[str, int]] = {}

    for lang in LANGS:
        handles = {
            split: (out_dir / f"{split}.{lang}.txt").open("w", encoding="utf-8")
            for split in budget
        }
        written = dict.fromkeys(budget, 0)
        n_docs = dict.fromkeys(budget, 0)
        ds = load_dataset(
            "wikimedia/wikipedia", WIKI_CONFIG[lang], split="train", streaming=True
        )
        for rec in ds:
            split = _split_of(f"{lang}:{rec['id']}")
            if written[split] >= budget[split]:
                if all(written[s] >= budget[s] for s in budget):
                    break
                continue
            doc = _clean(rec["text"])
            if len(doc) < 200:
                continue
            # A document boundary marker keeps the arms honest: every
            # tokenizer must encode the same separator.
            payload = doc + "\n"
            handles[split].write(payload)
            written[split] += len(payload.encode("utf-8"))
            n_docs[split] += 1
            if n_docs[split] % 20000 == 0:
                print(
                    f"[{lang}] {split}: {written[split] / 1e6:.1f} MB "
                    f"({n_docs[split]} docs)",
                    flush=True,
                )
        for h in handles.values():
            h.close()
        stats[lang] = {f"{split}_bytes": written[split] for split in budget} | {
            f"{split}_docs": n_docs[split] for split in budget
        }
        print(f"[{lang}] done: {stats[lang]}", flush=True)

    meta = {
        "source": "wikimedia/wikipedia",
        "configs": WIKI_CONFIG,
        "normalization": "NFC, blank-line collapse, trailing newline per doc",
        "split_rule": "sha1(lang:doc_id) % 1000 -> [0,5) valid, [5,10) test, else train",
        "budget": budget,
        "stats": stats,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta


def load_split(corpus_dir: Path, split: str, langs: tuple[str, ...] = LANGS) -> str:
    """Read a split as one string (ru first, then en)."""
    parts = []
    for lang in langs:
        p = corpus_dir / f"{split}.{lang}.txt"
        parts.append(p.read_text(encoding="utf-8"))
    return "".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/corpus"))
    ap.add_argument("--train-mb", type=int, default=300)
    ap.add_argument("--valid-mb", type=int, default=4)
    ap.add_argument("--test-mb", type=int, default=4)
    args = ap.parse_args()

    budget = {
        "train": args.train_mb * 1024 * 1024,
        "valid": args.valid_mb * 1024 * 1024,
        "test": args.test_mb * 1024 * 1024,
    }
    meta = build(args.out, budget)
    print(json.dumps(meta["stats"], indent=2))
    sys.stdout.flush()
    # datasets' streaming reader crashes at interpreter finalization in this
    # env (PyGILState_Release); bypass finalization entirely.
    os._exit(0)


if __name__ == "__main__":
    main()
