"""Perturbation suites: the robustness axis of the study.

Two families, kept separate because they probe different mechanisms:

* **case** -- pure surface casing. A factorized tokenizer should be nearly
  immune (case moved to a feature head); a BPE should degrade badly, because
  ``ПРИВЕТ`` and ``привет`` share no vocabulary entries.
* **typo** -- character-level corruption. This is *not* addressed by the
  case/suffix factorization; it is what the character n-gram bag in the model
  is for. Reported separately so the two claims never get conflated.

All functions are deterministic given ``seed`` and operate on letters only,
so document structure and whitespace stay intact.
"""

from __future__ import annotations

import random
import unicodedata

# Physically adjacent keys, used so substitutions look like real typing errors
# rather than uniform noise.
_QWERTY = [
    "qwertyuiop",
    "asdfghjkl",
    "zxcvbnm",
]
_JCUKEN = [
    "йцукенгшщзхъ",
    "фывапролджэ",
    "ячсмитьбю",
]


def _neighbors(rows: list[str]) -> dict[str, str]:
    out: dict[str, list[str]] = {}
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            near = []
            for dr in (-1, 0, 1):
                rr = r + dr
                if not 0 <= rr < len(rows):
                    continue
                for dc in (-1, 0, 1):
                    cc = c + dc
                    if dr == 0 and dc == 0:
                        continue
                    if 0 <= cc < len(rows[rr]):
                        near.append(rows[rr][cc])
            out[ch] = near
    return {k: "".join(v) for k, v in out.items()}


_KEY_NEIGHBORS: dict[str, str] = {}
_KEY_NEIGHBORS.update(_neighbors(_QWERTY))
_KEY_NEIGHBORS.update(_neighbors(_JCUKEN))


def _restore_case(src: str, ch: str) -> str:
    return ch.upper() if src.isupper() else ch


def typo(text: str, rate: float, seed: int = 0) -> str:
    """Corrupt a fraction ``rate`` of letters with realistic typing errors.

    Four error types in equal proportion: keyboard-adjacent substitution,
    deletion, transposition with the next character, and doubling.
    """
    rng = random.Random(seed)
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch.isalpha() and rng.random() < rate:
            kind = rng.randrange(4)
            low = ch.lower()
            if kind == 0 and low in _KEY_NEIGHBORS:
                out.append(_restore_case(ch, rng.choice(_KEY_NEIGHBORS[low])))
            elif kind == 1:
                pass  # deletion
            elif kind == 2 and i + 1 < n and text[i + 1].isalpha():
                out.append(text[i + 1])
                out.append(ch)
                i += 2
                continue
            else:
                out.append(ch)
                out.append(ch)  # doubling
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def case_lower(text: str, seed: int = 0) -> str:
    return text.lower()


def case_upper(text: str, seed: int = 0) -> str:
    return text.upper()


def case_title(text: str, seed: int = 0) -> str:
    out: list[str] = []
    prev_alpha = False
    for ch in text:
        if ch.isalpha():
            out.append(ch.lower() if prev_alpha else ch.upper())
            prev_alpha = True
        else:
            out.append(ch)
            prev_alpha = False
    return "".join(out)


def case_random(text: str, seed: int = 0, rate: float = 0.3) -> str:
    rng = random.Random(seed)
    return "".join(
        (ch.upper() if rng.random() < 0.5 else ch.lower())
        if ch.isalpha() and rng.random() < rate
        else ch
        for ch in text
    )


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


# name -> callable(text, seed) -> text
SUITES = {
    "clean": lambda t, seed=0: t,
    "typo1": lambda t, seed=0: typo(t, 0.01, seed),
    "typo5": lambda t, seed=0: typo(t, 0.05, seed),
    "typo10": lambda t, seed=0: typo(t, 0.10, seed),
    "lower": case_lower,
    "upper": case_upper,
    "title": case_title,
    "randcase": case_random,
}

CASE_SUITES = ("lower", "upper", "title", "randcase")
TYPO_SUITES = ("typo1", "typo5", "typo10")


def apply_suite(name: str, text: str, seed: int = 0) -> str:
    """Apply a suite and re-normalize, so every arm still sees NFC bytes."""
    return nfc(SUITES[name](text, seed))
