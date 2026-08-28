"""Interpretable per-token features — the routing signal, and later the genome.

Stage 1 routes tokens to shared parameters by *what they look like and how they
behave in the corpus*, not by where their pretrained embedding happens to sit.
That is the whole difference from stage 0, which fit structure to the embedding
matrix offline and cost +1.37 BPB at minimum: matching the matrix is not
matching the function. Features computed here decide only the **partition**;
the vectors at each leaf are re-learned through the frozen body.

This module is also the artifact the evolution phase mutates — ``token_features``
is the genome. That makes the *interface* more load-bearing than any particular
feature in it:

* **groups are first class.** ``ladder-calibration`` needs feature-group
  ablations, and the plan requires deliberately-bad anchors like single-feature
  and length-only routing. ``TokenFeatures.select`` and ``.without`` make those
  one call rather than a re-extraction.
* **no hidden state, no global caches, deterministic.** An LLM-mutated variant
  has to be droppable in and reproducible; a feature that silently depended on
  call order would make evolution's fitness signal noise.
* **every column is named.** An evolved routing program that nobody can read
  defeats the point of routing on interpretable features in the first place.

Groups, from the pre-registered "Feature set v0":

``surface``        script class, case shape, byte length, leading-space and
                   digit/punct content flags
``decomposition``  how the host tokenizer splits the surface: piece count, and
                   one-hot buckets of the first and last piece id
``ngram``          hashed character 3-5-gram presence, reusing ``model.hash_str``
                   (FNV-1a) rather than inventing a second hash
``corpus``         from the cached token stream: log unigram frequency, left and
                   right neighbour entropy, top-neighbour PMI, and burstiness

Study 1 killed character n-grams as an *additive input embedding* under an
emission-dominated metric. Here they are a *routing* signal deciding which
tokens share parameters, which is a different role, so that result does not
transfer — but it is why they are their own ablatable group rather than mixed
into ``surface``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import unicodedata

import numpy as np

from experiments.vocab_factorization.configs import FeatureSpec
from experiments.vocab_factorization.model import hash_str
from experiments.vocab_factorization.tokenization import piece_string

FEATURE_GROUPS: tuple[str, ...] = ("surface", "decomposition", "ngram", "corpus")


@dataclass(frozen=True)
class TokenFeatures:
    """``(V, F)`` float32 matrix plus the names and group of every column."""

    matrix: np.ndarray
    names: tuple[str, ...]
    groups: tuple[str, ...]  # parallel to names

    def __post_init__(self) -> None:
        if self.matrix.shape[1] != len(self.names) != len(self.groups):
            raise ValueError("matrix width, names and groups disagree")

    @property
    def group_names(self) -> tuple[str, ...]:
        seen = {}
        for g in self.groups:
            seen[g] = None
        return tuple(seen)

    def select(self, *groups: str) -> TokenFeatures:
        """Keep only these groups — the ablation primitive calibration needs."""
        unknown = set(groups) - set(self.group_names)
        if unknown:
            raise KeyError(f"unknown feature group(s) {sorted(unknown)}")
        keep = np.array([g in groups for g in self.groups])
        return TokenFeatures(
            matrix=self.matrix[:, keep],
            names=tuple(n for n, k in zip(self.names, keep, strict=True) if k),
            groups=tuple(g for g, k in zip(self.groups, keep, strict=True) if k),
        )

    def without(self, *groups: str) -> TokenFeatures:
        return self.select(*[g for g in self.group_names if g not in groups])

    def columns(self, *names: str) -> TokenFeatures:
        """Keep named columns — for single-feature and length-only anchors."""
        keep = np.array([n in names for n in self.names])
        if not keep.any():
            raise KeyError(f"none of {names} are columns")
        return TokenFeatures(
            matrix=self.matrix[:, keep],
            names=tuple(n for n, k in zip(self.names, keep, strict=True) if k),
            groups=tuple(g for g, k in zip(self.groups, keep, strict=True) if k),
        )


# --- surface --------------------------------------------------------------

_SCRIPTS = ("latin", "cyrillic", "digit", "punct", "other")
_CASES = ("lower", "title", "upper", "mixed", "caseless")


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


def _surface(surfaces: list[str]) -> tuple[np.ndarray, list[str]]:
    names = (
        [f"script.{s}" for s in _SCRIPTS]
        + [f"case.{c}" for c in _CASES]
        + [
            "len.bytes.log",
            "len.chars.log",
            "flag.leading_space",
            "flag.has_digit",
            "flag.has_punct",
            "flag.has_inner_space",
            "flag.is_ascii",
        ]
    )
    out = np.zeros((len(surfaces), len(names)), dtype=np.float32)
    for i, s in enumerate(surfaces):
        core = s.strip()
        # The dominant script of the stripped surface: one character decides it
        # for most tokens, and a mixed token lands in whichever script leads.
        if core:
            counts: dict[str, int] = {}
            for ch in core:
                k = _script_of(ch)
                counts[k] = counts.get(k, 0) + 1
            out[i, _SCRIPTS.index(max(counts, key=lambda k: counts[k]))] = 1.0
        else:
            out[i, _SCRIPTS.index("other")] = 1.0
        out[i, len(_SCRIPTS) + _CASES.index(_case_of(s))] = 1.0
        base = len(_SCRIPTS) + len(_CASES)
        out[i, base + 0] = np.log1p(len(s.encode("utf-8")))
        out[i, base + 1] = np.log1p(len(s))
        out[i, base + 2] = float(s[:1].isspace())
        out[i, base + 3] = float(any(c.isdigit() for c in s))
        out[i, base + 4] = float(any(not c.isalnum() and not c.isspace() for c in s))
        out[i, base + 5] = float(any(c.isspace() for c in core))
        out[i, base + 6] = float(s.isascii())
    return out, names


# --- decomposition --------------------------------------------------------


def _decomposition(
    surfaces: list[str], host_tok, buckets: int
) -> tuple[np.ndarray, list[str]]:
    names = ["pieces.count.log", "pieces.empty"]
    names += [f"pieces.first.b{i}" for i in range(buckets)]
    names += [f"pieces.last.b{i}" for i in range(buckets)]
    out = np.zeros((len(surfaces), len(names)), dtype=np.float32)
    for i, s in enumerate(surfaces):
        ids = host_tok(s, add_special_tokens=False)["input_ids"] if s else []
        if not ids:
            out[i, 1] = 1.0
            continue
        out[i, 0] = np.log1p(len(ids))
        out[i, 2 + (ids[0] % buckets)] = 1.0
        out[i, 2 + buckets + (ids[-1] % buckets)] = 1.0
    return out, names


# --- character n-grams ----------------------------------------------------


def _ngrams(
    surfaces: list[str], buckets: int, n_min: int, n_max: int
) -> tuple[np.ndarray, list[str]]:
    """Presence, not count: routing asks "does this token contain X", and a
    count would mostly re-encode length, which ``surface`` already carries."""
    names = [f"ngram.b{i}" for i in range(buckets)]
    out = np.zeros((len(surfaces), buckets), dtype=np.float32)
    for i, s in enumerate(surfaces):
        padded = f"<{s}>"
        for n in range(n_min, n_max + 1):
            for j in range(len(padded) - n + 1):
                out[i, hash_str(padded[j : j + n]) % buckets] = 1.0
    return out, names


# --- corpus statistics ----------------------------------------------------


def corpus_stats(stream: np.ndarray, vocab: int) -> dict[str, np.ndarray]:
    """Unigram, neighbour-entropy, PMI and burstiness over a token stream.

    Everything is computed in one pass over the bigram pairs with ``np.unique``
    rather than a per-token loop: the qwen stream is 53M tokens, and the
    difference is seconds against tens of minutes.
    """
    s = stream.astype(np.int64)
    n = s.size
    counts = np.bincount(s, minlength=vocab).astype(np.float64)
    total = max(1.0, counts.sum())

    stats = {"corpus.logfreq": np.log1p(counts)}
    pair = s[:-1] * np.int64(vocab) + s[1:]
    uniq, cnt = np.unique(pair, return_counts=True)
    left, right = uniq // vocab, uniq % vocab

    for side, idx, other in (("right", left, right), ("left", right, left)):
        # H(next | this): how predictable the token's neighbourhood is. A token
        # that is always followed by the same thing is doing a different job
        # from one that precedes anything, and that difference is invisible to
        # both the surface string and the raw frequency.
        denom = np.bincount(idx, weights=cnt, minlength=vocab)
        p = cnt / np.maximum(1e-12, denom[idx])
        ent = np.bincount(
            idx, weights=-p * np.log(np.maximum(p, 1e-12)), minlength=vocab
        )
        stats[f"corpus.entropy.{side}"] = ent
        del denom, p, ent
        if side == "right":
            # PMI of the single most frequent continuation, a cheap proxy for
            # "is this token part of a fixed collocation".
            best = np.zeros(vocab)
            order = np.argsort(cnt)[::-1]
            seen = np.zeros(vocab, dtype=bool)
            for k in order:
                i = left[k]
                if seen[i]:
                    continue
                seen[i] = True
                p_xy = cnt[k] / max(1.0, cnt.sum())
                p_x = counts[i] / total
                p_y = counts[right[k]] / total
                best[i] = np.log(max(p_xy, 1e-12) / max(p_x * p_y, 1e-12))
            stats["corpus.pmi.top"] = best

    # Burstiness: a token concentrated in a few places behaves differently from
    # one spread evenly, at the same total frequency.
    chunks = 256
    edges = np.linspace(0, n, chunks + 1).astype(np.int64)
    present = np.zeros(vocab, dtype=np.float64)
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        present += np.bincount(s[a:b], minlength=vocab) > 0
    expected = chunks * (1.0 - np.exp(-counts / max(1.0, chunks)))
    stats["corpus.burstiness"] = present / np.maximum(1.0, expected)
    stats["corpus.docfreq"] = present / chunks
    return stats


def _corpus(stats: dict[str, np.ndarray], vocab: int) -> tuple[np.ndarray, list[str]]:
    names = sorted(stats)
    out = np.stack([stats[k][:vocab] for k in names], axis=1).astype(np.float32)
    return out, names


# --- the entry point ------------------------------------------------------


def extract(
    arm,
    spec: FeatureSpec = FeatureSpec(),
    host_tok=None,
    stream: np.ndarray | None = None,
) -> TokenFeatures:
    """Features for every id in ``arm``'s stem vocabulary.

    ``host_tok`` is needed only for ``decomposition`` and ``stream`` only for
    ``corpus``; a spec that omits those groups needs neither, which is what
    makes the cheap ablations cheap to run.
    """
    vocab = arm.heads["stem"]
    surfaces = [piece_string(arm, i) for i in range(vocab)]
    blocks: list[np.ndarray] = []
    names: list[str] = []
    groups: list[str] = []

    def add(group: str, mat: np.ndarray, cols: list[str]) -> None:
        blocks.append(mat)
        names.extend(cols)
        groups.extend([group] * len(cols))

    for group in spec.groups:
        if group == "surface":
            add("surface", *_surface(surfaces))
        elif group == "decomposition":
            if host_tok is None:
                raise ValueError("the 'decomposition' group needs host_tok")
            add(
                "decomposition", *_decomposition(surfaces, host_tok, spec.piece_buckets)
            )
        elif group == "ngram":
            add(
                "ngram",
                *_ngrams(surfaces, spec.ngram_buckets, spec.ngram_min, spec.ngram_max),
            )
        elif group == "corpus":
            if stream is None:
                raise ValueError("the 'corpus' group needs a token stream")
            add("corpus", *_corpus(corpus_stats(stream, vocab), vocab))
        else:
            raise KeyError(f"unknown feature group {group!r}")

    return TokenFeatures(
        matrix=np.concatenate(blocks, axis=1).astype(np.float32),
        names=tuple(names),
        groups=tuple(groups),
    )


def cheap_spec(spec: FeatureSpec = FeatureSpec()) -> FeatureSpec:
    """Surface-only: no host tokenizer, no corpus pass. Useful for L0/L1."""
    return replace(spec, groups=("surface",))
