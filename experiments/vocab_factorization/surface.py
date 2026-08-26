"""Reversible surface-form factorization: text <-> (spans, case, space).

This module owns the *lossless* part of the pipeline. It splits text into
spans, and for letter-spans pulls out two orthogonal features:

    leading-space  (2 values)  -- kills the GPT-2 "token"/" token" duplication
    case class     (4 values)  -- LOWER / TITLE / UPPER / OTHER

The core invariant, checked per word rather than assumed:

    restore_case(case, core.lower()) == core

If a word fails that check (German ``STRAßE``, Turkish dotted i, and other
casing corner cases), it is classified OTHER and its exact characters are kept
verbatim, so the byte-fallback path handles it. Invertibility is therefore a
structural property of the segmentation, not a hope about Unicode.
"""

from __future__ import annotations

from dataclasses import dataclass

import regex

# GPT-2's pre-tokenizer shape, minus the English-only contraction rules
# (the corpus is ru+en) and with digits split individually.
#
# The ``\s+(?!\S)|\s+`` tail is the trick that leaves exactly one space
# attached to a following word: a whitespace run before a word matches the
# second alternative one character short, so " word" stays a single span.
_PAT = regex.compile(
    r" ?\p{L}+| ?\p{N}| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
    regex.UNICODE,
)

CASE_LOWER, CASE_TITLE, CASE_UPPER, CASE_OTHER = 0, 1, 2, 3
CASE_NAMES = ("lower", "title", "upper", "other")
N_CASE = 4
N_SPACE = 2

KIND_WORD = "word"
KIND_OTHER = "other"


@dataclass(slots=True)
class Span:
    """One atom of text.

    ``text`` is always the exact original substring, so ``"".join(s.text for s
    in segment(x)) == x`` holds unconditionally.

    For ``kind == KIND_WORD``: ``core`` is the casefolded letters and
    ``space``/``case`` are the extracted features. For ``KIND_OTHER`` the
    features are meaningless and ``core`` equals ``text``.
    """

    text: str
    kind: str
    space: int = 0
    case: int = CASE_LOWER
    core: str = ""


def classify_case(word: str) -> int:
    """Return the case class of a word, or CASE_OTHER if it is not restorable."""
    if word.islower():
        cls = CASE_LOWER
    elif word.isupper() and len(word) > 1:
        cls = CASE_UPPER
    elif word[:1].isupper() and (len(word) == 1 or word[1:].islower()):
        cls = CASE_TITLE
    else:
        return CASE_OTHER
    # Verify rather than trust: .lower() is not universally invertible.
    if restore_case(cls, word.lower()) != word:
        return CASE_OTHER
    return cls


def restore_case(case: int, lowered: str) -> str:
    if case == CASE_LOWER:
        return lowered
    if case == CASE_TITLE:
        return lowered[:1].upper() + lowered[1:]
    if case == CASE_UPPER:
        return lowered.upper()
    return lowered


def segment(text: str) -> list[Span]:
    """Split text into feature-annotated spans that concatenate back exactly."""
    spans: list[Span] = []
    for m in _PAT.finditer(text):
        raw = m.group()
        space = 1 if raw.startswith(" ") else 0
        body = raw[1:] if space else raw
        if body and body[0].isalpha():
            case = classify_case(body)
            if case == CASE_OTHER:
                # Not restorable from the lowercase form -- keep verbatim.
                spans.append(
                    Span(
                        text=raw,
                        kind=KIND_OTHER,
                        space=space,
                        case=CASE_OTHER,
                        core=body,
                    )
                )
            else:
                spans.append(
                    Span(
                        text=raw,
                        kind=KIND_WORD,
                        space=space,
                        case=case,
                        core=body.lower(),
                    )
                )
        else:
            spans.append(
                Span(text=raw, kind=KIND_OTHER, space=space, case=CASE_OTHER, core=body)
            )
    return spans


def desegment(spans: list[Span]) -> str:
    return "".join(s.text for s in spans)


def rebuild_word(space: int, case: int, core: str) -> str:
    """Inverse of the word path: features + casefolded core -> original text."""
    return (" " if space else "") + restore_case(case, core)


def check_roundtrip(text: str) -> None:
    """Raise if segmentation is not byte-exact. Cheap enough to run per corpus."""
    spans = segment(text)
    if desegment(spans) != text:
        raise AssertionError("segmentation is not lossless")
    for s in spans:
        if s.kind == KIND_WORD and rebuild_word(s.space, s.case, s.core) != s.text:
            raise AssertionError(f"word rebuild failed for {s.text!r}")
