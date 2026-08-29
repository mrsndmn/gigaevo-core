"""Seed program: surface features only.

Deliberately weaker than the full baseline extractor. Seeding with the strongest
known program makes the archive's first generation its best one and turns the
search into a random walk around a local optimum. This seeds with the cheapest
group that plainly carries signal — script, case, length, a few flags — and
leaves decomposition, character n-grams and corpus statistics for evolution to
rediscover or improve on.

Every column is named, because a routing program nobody can read is not the
thing this study is trying to produce.
"""

from __future__ import annotations

import unicodedata

import numpy as np

SCRIPTS = ("latin", "cyrillic", "digit", "punct", "other")
CASES = ("lower", "title", "upper", "mixed", "caseless")


def _script_of(ch: str) -> str:
    if ch.isdigit():
        return "digit"
    if not ch.isalpha():
        return "punct" if not ch.isspace() else "other"
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return "other"
    if "CYRILLIC" in name:
        return "cyrillic"
    return "latin" if "LATIN" in name else "other"


def _case_of(s: str) -> str:
    core = s.strip()
    if not core or not any(c.isalpha() for c in core):
        return "caseless"
    if core.islower():
        return "lower"
    if core.isupper():
        return "upper"
    if core[:1].isupper() and core[1:].islower():
        return "title"
    return "mixed"


def extract(arm, host_tok, stream):
    from experiments.vocab_factorization.tokenization import piece_string

    vocab = arm.heads["stem"]
    names = (
        [f"script.{s}" for s in SCRIPTS]
        + [f"case.{c}" for c in CASES]
        + ["len.bytes.log", "flag.leading_space", "flag.has_digit", "flag.has_punct"]
    )
    out = np.zeros((vocab, len(names)), dtype=np.float32)
    for i in range(vocab):
        s = piece_string(arm, i)
        core = s.strip()
        if core:
            counts: dict[str, int] = {}
            for ch in core:
                k = _script_of(ch)
                counts[k] = counts.get(k, 0) + 1
            out[i, SCRIPTS.index(max(counts, key=lambda k: counts[k]))] = 1.0
        else:
            out[i, SCRIPTS.index("other")] = 1.0
        out[i, len(SCRIPTS) + CASES.index(_case_of(s))] = 1.0
        base = len(SCRIPTS) + len(CASES)
        out[i, base + 0] = np.log1p(len(s.encode("utf-8")))
        out[i, base + 1] = float(s[:1].isspace())
        out[i, base + 2] = float(any(c.isdigit() for c in s))
        out[i, base + 3] = float(any(not c.isalnum() and not c.isspace() for c in s))
    return out, names


def entrypoint():
    return extract
