"""Progressive vocabulary shrink: interpretable rules over a *live* vocabulary.

Stage 0 killed one-shot replacement of the pretrained table (every offline
distillation cost at least +1.37 BPB against a 0.05 kill line). What the
evidence permits instead is a curriculum of small, named edits, each healed by
a short train. This module is the edits; ``train_transplant`` is the healing.

**A rule deletes rows without deleting information.** Merging " The", "the" and
"THE" into one row is only legitimate if the distinction they carried is moved
somewhere the loss still pays for it — here a *role* head, exactly the
factorization study 1 measured on tokenizers built from scratch, now applied to
a vocabulary that already has pretrained rows. Total NLL is summed over every
head (``train.eval_bpb``), so a merge that hides case in a side head buys
nothing. The gate is therefore an honest one: BPB within +0.01 of the masked
native reference **0.7796** (W0, ``results/transplant/orig-frozen.json``).

**Everything is a rewrite of (stem, role) pairs.** A vocabulary is a tuple of
stem byte strings plus the roles it is allowed to carry; ``Rewrite`` maps every
``(stem, role)`` of one vocabulary to a *sequence* of pairs in the next. That
one type covers all three rules: rule 0 and rule 1 emit exactly one pair (a
relabeling, so the token count is unchanged), rule 2 emits one or more (a
re-piecing, which lengthens the stream). Composition is substitution, so the
curriculum's x-axis — vocabulary size — and its parameter accounting are
*derived* from the mapping rather than asserted beside it.

**Bytes, not str.** ``tokenization.piece_string`` decodes an id with the HF
tokenizer, which replaces a byte-level BPE fragment with U+FFFD: host id 100 is
the single byte 0xA7 and comes back as "\\ufffd". A rule that canonicalised
*that* would silently corrupt the text it claims to re-spell. Surfaces here are
the raw bytes recovered from the byte-level alphabet, and a fragment that is
not valid UTF-8 simply has no case role — it is carried verbatim.

**The round trip is checked, not argued.** ``build`` verifies that every
rewritten pair still spells the same bytes it did before. That is what keeps
BPB comparable across the curve: the token stream changes, the byte stream it
encodes does not, and BPB's denominator is bytes.

**Merging ids is not the same as merging case.** Rule 1 rewrites ids that are
already case variants of one lexeme, and on clean text that is excellent — 0.7757
BPB, 0.0039 *better* than the pretrained model, at no cost in tokens. On
uppercase text it is a disaster: 1.9205, and 0.886 of the +1.145 lives in the
*stem* head. The reason is that the host tokenizer fragments uppercase text into
different, rarer pieces **before** any merge applies, so the merge fires on the
wrong fragments — 92.9% of ``upper``'s tokens do land in a lexeme group, and the
stem streams rule 1 leaves for ``upper`` and ``lower`` still share only 36.9% of
their unigram mass. ``case_fold`` + ``fold_encode`` is the repair: tokenize the
lowercased text, read the role off the original casing. That takes the same
comparison to **99.85%** and cuts ``upper`` from 1,573,813 tokens to 962,723.

Measured on Qwen3-1.7B's vocabulary and the real 230 MB stream (run the module,
2026-08-30; 26 s of CPU for the deepest rung once the stream is encoded):

=====================================  ======  ========  ====================
rule 0  active-vocab mask               74,957  153.51M   W0's mask, as deletion
rule 1  case/space -> stem + role bit   48,703   99.77M   token count unchanged
rule 2  re-piece the tail (< 100)       17,385   35.63M   +1.81% tokens
rule 2  re-piece the tail (< 1000)       4,783    9.82M   +14.07% tokens
-------------------------------------  ------  --------  --------------------
rule 0  mask, over the folded ids       58,587  119.99M   fewer ids are used
rule 1' case fold at tokenization       45,742   93.70M   -0.08% tokens
rule 2  re-piece the tail (< 100)       16,632   34.09M   +1.56% tokens
rule 2  re-piece the tail (< 1000)       4,890   10.04M   +12.82% tokens
=====================================  ======  ========  ====================

Run::

    PYTHONPATH=. python -m experiments.vocab_factorization.shrink
    PYTHONPATH=. python -m experiments.vocab_factorization.shrink --curriculum fold
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from experiments.vocab_factorization.configs import ShrinkRule

# --- the role algebra -----------------------------------------------------
#
# A role is a (leading space, case) pair: 3 case forms x 2 space states. Six
# values is not a design target, it is what the merge rule can *invert* — see
# ``_cased``, which refuses any transform that does not round-trip.

ROLE_NAMES: tuple[str, ...] = (
    "lower",
    "title",
    "upper",
    "sp+lower",
    "sp+title",
    "sp+upper",
)
N_ROLES = len(ROLE_NAMES)


def _cased(stem: bytes, case: int) -> bytes | None:
    """The ``case`` form of ``stem``, or None if the form is not legal.

    Two ways to be illegal, both load-bearing. A form that does not lowercase
    back to the stem is not invertible — 'ß'.upper() is 'SS' — so accepting it
    would let a merge change the text. And a form *equal* to the stem is a
    duplicate spelling (b'42'.upper() is b'42'), which would make two roles emit
    the same bytes and waste a distinction the loss has to pay for.
    """
    if case == 0:
        return stem
    try:
        s = stem.decode("utf-8")
    except UnicodeDecodeError:
        return None  # a fragment of a multi-byte character has no case
    form = s[:1].upper() + s[1:] if case == 1 else s.upper()
    if form == s or form.lower() != s:
        return None
    return form.encode("utf-8")


def _canonicalize(surface: bytes) -> tuple[bytes, int] | None:
    """``surface`` as (lowercase stem, role), or None if the rule cannot apply."""
    if not surface:
        return None
    space = surface.startswith(b" ")
    core = surface[1:] if space else surface
    # A stem that itself begins with a space would make b" x" spellable twice —
    # as (b" x", no space) and as (b"x", space) — so a multi-space surface stays
    # verbatim rather than becoming an ambiguous stem.
    if not core or core.startswith(b" "):
        return None
    try:
        low = core.decode("utf-8").lower().encode("utf-8")
    except UnicodeDecodeError:
        return None
    for case in range(3):
        if _cased(low, case) == core:
            return low, case + 3 * int(space)
    return None  # mixed case ("aBc"): not expressible as one role


@dataclass(frozen=True)
class Vocabulary:
    """Stem byte strings plus the number of roles each may carry.

    ``n_roles == 1`` is a vocabulary with no role head at all — the host's own
    table, and everything before rule 1.
    """

    stems: tuple[bytes, ...]
    n_roles: int = 1

    def __len__(self) -> int:
        return len(self.stems)

    def spell(self, stem: int, role: int = 0) -> bytes | None:
        """The bytes this pair emits, or None if the pair is not legal."""
        if role >= self.n_roles:
            return None
        core = _cased(self.stems[stem], role % 3)
        if core is None:
            return None
        return (b" " if role >= 3 else b"") + core

    def spellings(self, keep: set[int] | None = None) -> dict[bytes, tuple[int, int]]:
        """Every legal spelling -> the pair that produces it.

        This is the alphabet rule 2 re-pieces *into*: after rule 1 the stems are
        lowercase, so spelling b" Comp" means (stem b"comp", role sp+title). A
        table of stems alone would be unable to re-piece anything cased.
        """
        out: dict[bytes, tuple[int, int]] = {}
        for i in range(len(self.stems)) if keep is None else sorted(keep):
            for r in range(self.n_roles):
                b = self.spell(i, r)
                if b:
                    out.setdefault(b, (i, r))
        return out


# --- rewrites -------------------------------------------------------------


@dataclass(frozen=True)
class Rewrite:
    """(src stem, src role) -> a sequence of (dst stem, dst role), CSR.

    Keyed by ``stem * src_roles + role``. A pair with an empty sequence is one
    the rule considers unreachable; ``apply`` raises on it by name rather than
    dropping the token, because a silently shortened stream would still produce
    a plausible-looking BPB.
    """

    src_roles: int
    dst: Vocabulary
    offsets: np.ndarray  # (n_src_pairs + 1,) int64
    stems: np.ndarray  # int32
    roles: np.ndarray  # int8

    @property
    def n_src_pairs(self) -> int:
        return int(self.offsets.size - 1)

    def pairs(self, stem: int, role: int = 0) -> list[tuple[int, int]]:
        k = stem * self.src_roles + role
        a, b = int(self.offsets[k]), int(self.offsets[k + 1])
        return list(
            zip(self.stems[a:b].tolist(), self.roles[a:b].tolist(), strict=True)
        )

    def _expand(self, key: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        start = self.offsets[key]
        length = self.offsets[key + 1] - start
        total = int(length.sum())
        # Ragged gather: element j of group g reads ``start[g] + (j - out[g])``.
        out = np.zeros(key.size + 1, dtype=np.int64)
        np.cumsum(length, out=out[1:])
        idx = np.repeat(start - out[:-1], length) + np.arange(total, dtype=np.int64)
        return self.stems[idx], self.roles[idx], length

    def apply(
        self, stems: np.ndarray, roles: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        stems = np.asarray(stems, dtype=np.int64)
        key = stems * self.src_roles
        if roles is not None:
            key = key + np.asarray(roles, dtype=np.int64)
        if key.size and int(key.max()) >= self.n_src_pairs:
            raise ValueError(
                f"token id {int(key.max()) // self.src_roles} is outside the "
                f"{self.n_src_pairs // self.src_roles}-symbol source vocabulary"
            )
        empty = self.offsets[key + 1] == self.offsets[key]
        if empty.any():
            bad = np.unique(key[empty])[:8] // self.src_roles
            raise ValueError(
                f"{int(empty.sum())} tokens use source symbols the shrink plan "
                f"maps to nothing, e.g. {bad.tolist()} — the plan was built "
                "over a different vocabulary than the stream it is being "
                "applied to"
            )
        s, r, _ = self._expand(key)
        return s, r

    def then(self, other: Rewrite) -> Rewrite:
        """Compose: this rewrite, then ``other``. Substitution, vectorized."""
        key = self.stems.astype(np.int64) * other.src_roles + self.roles
        stems, roles, length = other._expand(key)
        cum = np.zeros(length.size + 1, dtype=np.int64)
        np.cumsum(length, out=cum[1:])
        offsets = np.zeros(self.offsets.size, dtype=np.int64)
        offsets[1:] = cum[self.offsets[1:]]
        return Rewrite(self.src_roles, other.dst, offsets, stems, roles)


def _rewrite_from_lists(
    src_pairs: int, src_roles: int, dst: Vocabulary, images: dict[int, list]
) -> Rewrite:
    """Build a Rewrite from ``{src key: [(stem, role), ...]}``; missing = empty."""
    offsets = np.zeros(src_pairs + 1, dtype=np.int64)
    for k, v in images.items():
        offsets[k + 1] = len(v)
    np.cumsum(offsets, out=offsets)
    flat = [p for k in sorted(images) for p in images[k]]
    stems = np.array([p[0] for p in flat], dtype=np.int32)
    roles = np.array([p[1] for p in flat], dtype=np.int8)
    return Rewrite(src_roles, dst, offsets, stems, roles)


# --- the rules ------------------------------------------------------------


@dataclass(frozen=True)
class Step:
    """One curriculum rung: the rule, its rewrite, and what it actually did.

    ``case_invariant`` marks a rung whose vocabulary is meant to be *entered by
    re-tokenizing* (``fold_encode``) rather than by relabeling host ids. It
    changes what "reachable" means downstream — see ``_case_variants``.
    """

    rule: str
    params: dict
    rewrite: Rewrite
    stats: dict
    case_invariant: bool = False

    @property
    def n_stems(self) -> int:
        return len(self.rewrite.dst)


@dataclass(frozen=True)
class Context:
    """What a rule may look at besides the vocabulary it is shrinking.

    Two streams, and keeping them apart is load-bearing. ``train_*`` is the
    corpus, and it is what a *frequency* threshold may look at. ``reach_*`` is
    the active id set pushed through the rules so far, and it is what
    *reachability* means: 2,480 of Qwen3's active ids never occur in the train
    stream at all — they come from the perturbation suites (W0) — so a rule that
    decided what to keep by looking at the training corpus alone would leave
    those tokens unspellable and take the robustness suites off the table.
    """

    host: Vocabulary
    train_stems: np.ndarray
    train_roles: np.ndarray | None
    reach_stems: np.ndarray
    reach_roles: np.ndarray | None

    def reach_keys(self, n_roles: int) -> np.ndarray:
        key = np.asarray(self.reach_stems, dtype=np.int64) * n_roles
        if self.reach_roles is not None:
            key = key + np.asarray(self.reach_roles, dtype=np.int64)
        return np.unique(key)


def _rule_mask(vocab: Vocabulary, rule: ShrinkRule, ctx: Context) -> Step:
    """Rule 0 — W0's active-vocabulary mask, expressed as deletion.

    The mask and a shrink are different operations on the same object: the mask
    restricts the softmax over a table that keeps its rows, a shrink re-indexes
    the table and deletes them. At this rung they are numerically the same
    thing, which is the point — ``shrink0-mask`` must reproduce W0's 0.7796
    digit for digit, and that is the machinery's correctness check rather than a
    new measurement. Rule 0 is free on BPB by construction (masking only removes
    softmax competitors); the gate lives on rules 1+.
    """
    keep = np.unique(np.asarray(ctx.reach_stems, dtype=np.int64))
    dst = Vocabulary(tuple(vocab.stems[int(i)] for i in keep), n_roles=1)
    images = {int(i) * vocab.n_roles: [(n, 0)] for n, i in enumerate(keep)}
    rewrite = _rewrite_from_lists(
        len(vocab) * vocab.n_roles, vocab.n_roles, dst, images
    )
    return Step(
        rule.rule,
        {},
        rewrite,
        {"kept": len(dst), "dropped": len(vocab) - len(dst)},
    )


def _rule_case_space(vocab: Vocabulary, rule: ShrinkRule, ctx: Context) -> Step:
    """Rule 1 — merge case/space variants of one lexeme into stem + role bit.

    The fact factorization of study 1, applied to a vocabulary that already has
    pretrained rows instead of to a tokenizer trained from scratch. A group is
    merged only if it has ``min_group`` members, so the canonical row it
    collapses onto is emitted by at least that many tokens and no unreachable
    row is ever created — which is what lets the softmax mask be dropped
    entirely at this rung (see the module docstring and ``train_transplant``).

    Scope note: grouping over the *current* vocabulary rather than over the full
    host table is not a shortcut. Measured both ways on Qwen3 + this corpus, the
    resulting vocabulary is 48,703 either way; grouping over the current
    vocabulary additionally keeps every stem a real host surface wherever one
    exists, which is a strictly better starting point for the healing train.
    """
    if vocab.n_roles != 1:
        raise ValueError("case_space expects a role-free vocabulary")

    groups: dict[bytes, list[int]] = {}
    role_of: dict[int, int] = {}
    for i, s in enumerate(vocab.stems):
        c = _canonicalize(s)
        if c is None:
            continue
        groups.setdefault(c[0], []).append(i)
        role_of[i] = c[1]
    merged = {c: g for c, g in groups.items() if len(g) >= rule.min_group}
    member_of = {i: c for c, g in merged.items() for i in g}

    stems: list[bytes] = []
    index: dict[bytes, int] = {}
    images: dict[int, list] = {}
    for i, s in enumerate(vocab.stems):
        key = member_of.get(i)
        if key is None:
            key, role = s, 0
        else:
            role = role_of[i]
        j = index.get(key)
        if j is None:
            j = index[key] = len(stems)
            stems.append(key)
        images[i] = [(j, role)]

    dst = Vocabulary(tuple(stems), n_roles=N_ROLES if merged else 1)
    counts = np.bincount(ctx.train_stems, minlength=len(vocab))
    moved = int(sum(counts[i] for i in member_of if role_of[i] != 0))
    biggest = sorted(merged.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:8]
    return Step(
        rule.rule,
        {"min_group": rule.min_group},
        _rewrite_from_lists(len(vocab), 1, dst, images),
        {
            "groups": len(merged),
            "ids_merged": sum(len(g) for g in merged.values()),
            "ids_saved": len(vocab) - len(dst),
            "token_fraction_with_a_role": moved / max(1, ctx.train_stems.size),
            "examples": [
                [c.decode("utf-8", "replace")]
                + [vocab.stems[i].decode("utf-8", "replace") for i in g]
                for c, g in biggest
            ],
        },
    )


def _rule_case_fold(vocab: Vocabulary, rule: ShrinkRule, ctx: Context) -> Step:
    """Rule 1' — the same merged table, entered by re-tokenizing instead.

    Rule 1 merges *ids*. Measured (W2 diagnostic, 2026-08-30) that is not the
    same thing as merging case: the host tokenizer fragments uppercase text
    before any merge can apply, so ``upper`` and ``lower`` leave stem streams
    whose unigram multisets overlap only **36.9%** and whose lengths differ by
    63%. The merge fires — 92.9% of ``upper`` tokens land in a lexeme group —
    but on *different fragments*, and shrink1-case pays +0.886 BPB in the stem
    head for it. ``fold_encode`` produces the stream this vocabulary was
    designed for: tokenize the lowercased text, read the role off the original.

    The vocabulary is exactly rule 1's, with one change: ``min_group`` is forced
    to 1. That is not a tuning knob, it is what makes the encoder independent of
    the plan. The encoder decides "may I re-case this piece?" from the piece's
    own surface (``_canonicalize``); if a singleton group were left verbatim,
    a surface that canonicalizes would still be an unmerged row and imposing a
    case role on it would spell the wrong bytes. Row count is unaffected — two
    surfaces sharing a core would be a group of two, so a singleton's core
    collides with nothing and one row is traded for one row.
    """
    if rule.min_group != 1:
        raise ValueError(
            "case_fold requires min_group=1: the fold encoder reads the merge "
            "decision off a surface, not off the group's size"
        )
    step = _rule_case_space(vocab, replace(rule, min_group=1), ctx)
    stats = dict(step.stats)
    # Inherited from rule 1, where it counts tokens whose *id* carries a case or
    # space role. Under a fold the stream is lowercase, so the same number counts
    # leading spaces only and the case fraction lives in ``FoldEncoding.stats``.
    # Renaming it here stops the artifact from being read as the fold's own.
    stats["token_fraction_with_a_space"] = stats.pop("token_fraction_with_a_role")
    return Step("case_fold", {"min_group": 1}, step.rewrite, stats, True)


def _case_variants(
    stems: np.ndarray, roles: np.ndarray | None, vocab: Vocabulary
) -> tuple[np.ndarray, np.ndarray]:
    """Every legal case form of a reachable pair is reachable after a fold.

    Reachability is what rule 2 re-pieces over, and after a fold it is no longer
    "the pairs the active ids happen to spell". The point of folding is that any
    casing of a reachable word is emittable whether or not the perturbation
    suites happened to contain it, so the closure is taken over the role algebra
    rather than over the measured suites. Illegal forms (``b"42"`` has no upper)
    are dropped, so ``spell`` stays total on the result.
    """
    stems = np.asarray(stems, dtype=np.int64)
    base = (
        np.zeros_like(stems)
        if roles is None
        else (np.asarray(roles, dtype=np.int64) // 3) * 3
    )
    s = np.repeat(stems, 3)
    r = np.repeat(base, 3) + np.tile(np.arange(3, dtype=np.int64), stems.size)
    key = np.unique(s * vocab.n_roles + r)
    s, r = np.divmod(key, vocab.n_roles)
    keep = np.fromiter(
        (vocab.spell(int(a), int(b)) is not None for a, b in zip(s, r, strict=True)),
        dtype=bool,
        count=s.size,
    )
    return s[keep], r[keep]


def _greedy(
    text: bytes, table: dict[bytes, tuple[int, int]], span: dict[int, int]
) -> list[tuple[int, int]] | None:
    """Longest-match-first segmentation of ``text`` over ``table``."""
    out: list[tuple[int, int]] = []
    i, n = 0, len(text)
    while i < n:
        for ln in range(min(span.get(text[i], 1), n - i), 0, -1):
            hit = table.get(text[i : i + ln])
            if hit is not None:
                out.append(hit)
                i += ln
                break
        else:
            return None
    return out


def _rule_repiece_tail(vocab: Vocabulary, rule: ShrinkRule, ctx: Context) -> Step:
    """Rule 2 — spell the frequency tail out of the pieces that survive.

    A stem the corpus uses fewer than ``min_count`` times stops being a row and
    becomes a segmentation over the retained spellings. Nothing is lost: the
    concatenation is byte-identical (checked in ``build``), the stream just gets
    longer, and BPB's denominator is bytes.

    **The byte floor is a measured necessity, not a safety belt.** The active
    vocabulary is not closed under re-piecing: 66 of the 256 single-byte host
    tokens never occur in this ru+en corpus (control characters and UTF-8 lead
    bytes), so rule 0 deletes them and a tail stem containing one would have no
    segmentation at all. Rule 2 therefore reaches back to the host tokenizer for
    the stems needed to spell every byte — but adds only the ones a segmentation
    actually uses, so the "every row is reachable" invariant still holds and the
    accounting stays exact.
    """
    counts = np.bincount(ctx.train_stems, minlength=len(vocab))
    retained = {i for i in range(len(vocab)) if counts[i] >= rule.min_count}

    floor: dict[bytes, tuple[bytes, int]] = {}
    for s in ctx.host.stems:
        if len(s) != 1 or s in floor:
            continue
        c = _canonicalize(s)
        floor[s] = (c[0], c[1]) if (c and vocab.n_roles > 1) else (s, 0)
    if len(floor) != 256:
        raise ValueError(
            f"the host vocabulary spells only {len(floor)} of 256 bytes, so a "
            "re-piecing is not guaranteed to terminate"
        )

    # Spellings available to the segmenter: the retained stems, plus the floor
    # stems keyed by string because they may not be in this vocabulary at all.
    table: dict[bytes, tuple[bytes, int]] = {}
    for text, (i, r) in vocab.spellings(retained).items():
        table[text] = (vocab.stems[i], r)
    for text, pair in floor.items():
        table.setdefault(text, pair)
    span: dict[int, int] = {}
    for text in table:
        span[text[0]] = max(span.get(text[0], 1), len(text))

    # Only pairs the active set can actually reach are rewritten; the rest map
    # to nothing and ``Rewrite.apply`` names them if they ever turn up.
    reach = ctx.reach_keys(vocab.n_roles)
    segments: dict[int, list[tuple[bytes, int]]] = {}
    used: set[bytes] = set()
    for key in reach.tolist():
        i, r = divmod(key, vocab.n_roles)
        if i in retained:
            segments[key] = [(vocab.stems[i], r)]
            used.add(vocab.stems[i])
            continue
        text = vocab.spell(i, r)
        pieces = _greedy(text, table, span) if text is not None else None
        if pieces is None:
            raise ValueError(
                f"stem {vocab.stems[i]!r} role {r} has no segmentation over the "
                "retained vocabulary — the byte floor is incomplete"
            )
        segments[key] = pieces
        used.update(p[0] for p in pieces)

    # Only stems a segmentation actually uses become rows, so "every row of the
    # shrunk table is reachable" survives the byte floor and the accounting is
    # the table that exists rather than the table that was proposed.
    kept = [i for i in sorted(retained) if vocab.stems[i] in used]
    stems = [vocab.stems[i] for i in kept]
    revived = sorted(used - set(stems))
    stems += revived
    index = {s: n for n, s in enumerate(stems)}
    dst = Vocabulary(tuple(stems), n_roles=vocab.n_roles)
    images = {k: [(index[s], r) for s, r in v] for k, v in segments.items()}

    lengths = np.ones(len(vocab) * vocab.n_roles, dtype=np.int64)
    for k, v in images.items():
        lengths[k] = len(v)
    key = np.asarray(ctx.train_stems, dtype=np.int64) * vocab.n_roles
    if ctx.train_roles is not None:
        key = key + np.asarray(ctx.train_roles, dtype=np.int64)
    grown = int(lengths[key].sum())
    show = [k for k in sorted(segments) if len(segments[k]) > 1][:8]
    return Step(
        rule.rule,
        {"min_count": rule.min_count},
        _rewrite_from_lists(len(vocab) * vocab.n_roles, vocab.n_roles, dst, images),
        {
            "retained": len(retained),
            "dropped": len(vocab) - len(retained),
            "retained_unused": len(retained) - len(kept),
            "revived": len(revived),
            "token_inflation": grown / max(1, ctx.train_stems.size) - 1.0,
            "examples": [
                [vocab.spell(*divmod(k, vocab.n_roles)).decode("utf-8", "replace")]
                + [
                    (dst.spell(s, r) or b"?").decode("utf-8", "replace")
                    for s, r in images[k]
                ]
                for k in show
            ],
        },
    )


RULES = {
    "mask": _rule_mask,
    "case_space": _rule_case_space,
    "case_fold": _rule_case_fold,
    "repiece_tail": _rule_repiece_tail,
}


# --- case folding at tokenization time ------------------------------------
#
# Rule 1 rewrites ids that are already case variants of one lexeme. The
# diagnostic behind ``_rule_case_fold`` says uppercase text does not *emit*
# those ids in the first place — it is fragmented differently before any merge
# applies. So this half tokenizes the lowercased text and recovers the case from
# the original, which is what study 1's factorized arms did at the tokenizer
# (``tokenization.FactorizedArm``) and what rule 1 only approximates.


@dataclass(frozen=True)
class FoldEncoding:
    """Host ids over the *lowercased* text, plus the case each token restores.

    ``case`` is the case index (0/1/2) to impose on the id's role, or **-1** for
    "keep the role the id already carries". -1 rather than 0 is load-bearing: a
    token that came back through the fallback path spells the original text
    itself, and its id may well be an uppercase surface whose own role is
    ``upper`` — forcing 0 onto it would re-spell it in lowercase.
    """

    ids: np.ndarray  # int32
    case: np.ndarray  # int8
    nbytes: int
    stats: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.ids.size)


def _utf8_len(cp: np.ndarray) -> np.ndarray:
    n = np.ones(cp.size, dtype=np.int64)
    n += (cp >= 0x80).astype(np.int64)
    n += (cp >= 0x800).astype(np.int64)
    n += (cp >= 0x10000).astype(np.int64)
    return n


def _lower_map(text: str) -> tuple[str, bytes, np.ndarray, np.ndarray]:
    """Lowercase ``text`` and map the result's byte offsets back onto it.

    Returns ``(low, low_bytes, omap, changed)``. ``omap[p]`` is the byte offset
    in ``text`` that lowercase byte ``p`` came from, or -1 where the question
    has no answer; ``changed[p]`` counts lowercase bytes before ``p`` that
    belong to a character the lowering altered.

    **Lowercasing is done character by character, not with ``str.lower()``.**
    Not for speed — the vectorized path here is a codepoint LUT and is faster —
    but because ``str.lower()`` is context-sensitive (Greek final sigma) and
    length-changing ('İ' becomes two characters), so the whole-string result has
    no exact offset map. Per character it does, and the -1 entries mark the
    interiors of characters whose bytes moved, which is precisely where a
    byte-level BPE boundary would be unattributable.
    """
    cp = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)
    uniq, inv = np.unique(cp, return_inverse=True)
    inv = inv.reshape(-1)
    forms = [chr(int(c)).lower() for c in uniq.tolist()]
    same_u = np.array(
        [f == chr(int(c)) for c, f in zip(uniq.tolist(), forms, strict=True)],
        dtype=bool,
    )
    if all(len(f) == 1 for f in forms):
        lut = np.array([ord(f) for f in forms], dtype=np.uint32)
        low = lut[inv].astype("<u4").tobytes().decode("utf-32-le")
    else:
        low = "".join([forms[i] for i in inv.tolist()])
    low_bytes = low.encode("utf-8")

    blen = np.array([len(f.encode("utf-8")) for f in forms], dtype=np.int64)
    nb_low = blen[inv] if inv.size else np.zeros(0, dtype=np.int64)
    ob = np.zeros(cp.size + 1, dtype=np.int64)
    np.cumsum(_utf8_len(cp), out=ob[1:])
    lb = np.zeros(cp.size + 1, dtype=np.int64)
    np.cumsum(nb_low, out=lb[1:])
    same = same_u[inv] if inv.size else np.zeros(0, dtype=bool)

    n = int(lb[-1])
    omap = np.full(n + 1, -1, dtype=np.int64)
    pos = np.arange(n, dtype=np.int64)
    keep = np.repeat(same, nb_low)
    omap[pos[keep]] = pos[keep] + np.repeat(ob[:-1] - lb[:-1], nb_low)[keep]
    omap[lb[:-1]] = ob[:-1]
    omap[n] = int(ob[-1])
    changed = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.repeat(~same, nb_low), out=changed[1:])
    return low, low_bytes, omap, changed


def _fold_case(surface: bytes, original: bytes) -> int:
    """Which case role turns ``surface`` back into ``original``.

    0/1/2 for a case index, -1 for "the surface is already the original", -2 for
    "no role spells this" — the mixed-case answer, and the only path that costs
    anything.
    """
    if surface == original:
        return -1
    canon = _canonicalize(surface)
    if canon is None:
        return -2
    core, role = canon
    lead = b" " if role >= 3 else b""
    for c in (1, 2):
        form = _cased(core, c)
        if form is not None and lead + form == original:
            return c
    return -2


def fold_encode(text: str, tok, host: Vocabulary) -> FoldEncoding:
    """Tokenize the lowercased text; read each token's case off the original.

    The alignment is done in **bytes, not characters**. HF's
    ``return_offsets_mapping`` reports character spans, and a byte-level BPE
    splits characters: measured on this tokenizer, 'ǅ' comes back as two tokens
    both claiming the span (0, 1) and a combining mark is dropped from the map
    entirely. Byte-level BPE *is* an exact byte segmentation, so the token
    surfaces' own lengths give a partition that cannot be wrong, and
    ``_lower_map`` carries it back to the original text.

    Two escape hatches, both counted rather than assumed away:

    * a token whose span is not attributable (its boundary lands inside a
      character the lowering moved), and
    * a token no single role can re-case — the mixed-case case, ``"aBc"``.

    Both fall back to **tokenizing that region of the original text** and
    emitting it with the role its own ids carry, which is exactly what rule 1
    does. Study 1's answer here was per-run case classification
    (``configs.ArmSpec.piece_case``), which is unavailable to a transplant: its
    runs are new stems, and this vocabulary's rows are pretrained host rows. The
    fallback therefore adds **no row at all**, unlike study 1's verbatim path —
    what it costs is fragmentation, on the region that hits it.
    """
    raw = text.encode("utf-8")
    low, low_bytes, omap, changed = _lower_map(text)
    ids = np.asarray(tok(low, add_special_tokens=False)["input_ids"], dtype=np.int64)
    n = int(ids.size)
    slen = np.fromiter((len(s) for s in host.stems), np.int64, len(host.stems))
    bounds = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(slen[ids], out=bounds[1:])
    if int(bounds[-1]) != len(low_bytes):
        raise ValueError(
            f"the token surfaces spell {int(bounds[-1])} bytes but the "
            f"lowercased text has {len(low_bytes)} — this vocabulary is not "
            "the one the tokenizer emits"
        )

    rawb = np.frombuffer(raw, dtype=np.uint8)
    ob = omap[bounds]
    aligned = ob >= 0
    span_low = bounds[1:] - bounds[:-1]
    attributable = aligned[:-1] & aligned[1:]
    # Where the fallback is allowed to cut. A byte-level BPE splits characters
    # (' ПРИВЕТ' comes back as five tokens over six two-byte characters), so an
    # attributable boundary is not necessarily a character boundary — and the
    # fallback has to hand the tokenizer a *string*. Cutting only at lead bytes
    # is what keeps that decode total.
    cut = aligned.copy()
    inside = aligned & (ob < len(raw))
    cut[inside] = (rawb[ob[inside]] & 0xC0) != 0x80
    # A token is untouched by the lowering iff no altered character overlaps it
    # and its two spans have the same length; then its surface *is* the original
    # substring and no role has to be imposed.
    plain = (
        attributable
        & (changed[bounds[1:]] == changed[bounds[:-1]])
        & ((ob[1:] - ob[:-1]) == span_low)
    )
    if plain.any():
        lowb = np.frombuffer(low_bytes, dtype=np.uint8)
        covered = np.repeat(plain, span_low)
        src = np.arange(lowb.size, dtype=np.int64) + np.repeat(
            ob[:-1] - bounds[:-1], span_low
        )
        if not np.array_equal(lowb[covered], rawb[src[covered]]):
            raise ValueError(
                "case fold: a span the lowering left alone does not match the "
                "original text — the byte offset map is wrong"
            )

    case = np.full(n, -1, dtype=np.int8)
    fallback = ~attributable
    memo: dict[tuple[int, bytes], int] = {}
    for i in np.flatnonzero(attributable & ~plain).tolist():
        original = raw[int(ob[i]) : int(ob[i + 1])]
        key = (int(ids[i]), original)
        c = memo.get(key)
        if c is None:
            c = memo[key] = _fold_case(host.stems[int(ids[i])], original)
        if c == -2:
            fallback[i] = True
        else:
            case[i] = c

    # Grow each failing token out to the nearest cuttable boundaries, then merge
    # the runs that touch. Byte 0 and the final byte always cut, so the growth
    # terminates.
    groups: list[list[int]] = []
    for i in np.flatnonzero(fallback).tolist():
        a, b = i, i + 1
        while a > 0 and not cut[a]:
            a -= 1
        while b < n and not cut[b]:
            b += 1
        if groups and groups[-1][1] >= a:
            groups[-1][1] = max(groups[-1][1], b)
        else:
            groups.append([a, b])

    out_ids: list[np.ndarray] = []
    out_case: list[np.ndarray] = []
    cur = 0
    fb_tokens = fb_bytes = 0
    for a, b in groups:
        if a > cur:
            out_ids.append(ids[cur:a])
            out_case.append(case[cur:a])
        lo, hi = int(ob[a]), int(ob[b])
        sub = np.asarray(
            tok(raw[lo:hi].decode("utf-8"), add_special_tokens=False)["input_ids"],
            dtype=np.int64,
        )
        got = b"".join(host.stems[j] for j in sub.tolist())
        if got != raw[lo:hi]:
            raise ValueError(
                f"case fold: the fallback re-encoding of bytes [{lo}, {hi}) "
                f"spells {got[:40]!r}, not {raw[lo:hi][:40]!r}"
            )
        out_ids.append(sub)
        out_case.append(np.full(sub.size, -1, dtype=np.int8))
        fb_tokens += int(sub.size)
        fb_bytes += hi - lo
        cur = b
    if cur < n:
        out_ids.append(ids[cur:])
        out_case.append(case[cur:])

    final_ids = (
        np.concatenate(out_ids) if out_ids else np.zeros(0, dtype=np.int64)
    ).astype(np.int32)
    final_case = np.concatenate(out_case) if out_case else np.zeros(0, dtype=np.int8)
    return FoldEncoding(
        final_ids,
        final_case,
        len(raw),
        {
            "tokens": int(final_ids.size),
            "lowercase_tokens": n,
            "recased_tokens": int((final_case >= 0).sum()),
            "fallback_groups": len(groups),
            "fallback_tokens": fb_tokens,
            "fallback_bytes": fb_bytes,
            "unattributable_boundaries": int((~aligned).sum()),
        },
    )


def fold_stats(encs: list[FoldEncoding]) -> dict:
    """Sum the per-chunk counters and add the rates worth reporting."""
    keys = (
        "tokens",
        "lowercase_tokens",
        "recased_tokens",
        "fallback_groups",
        "fallback_tokens",
        "fallback_bytes",
        "unattributable_boundaries",
    )
    out = {k: sum(int(e.stats.get(k, 0)) for e in encs) for k in keys}
    nb = sum(e.nbytes for e in encs)
    out["bytes"] = nb
    out["fallback_token_fraction"] = out["fallback_tokens"] / max(1, out["tokens"])
    out["fallback_byte_fraction"] = out["fallback_bytes"] / max(1, nb)
    out["recased_fraction"] = out["recased_tokens"] / max(1, out["tokens"])
    return out


def verify_fold(
    plan: ShrinkPlan, arrays: dict, raw: bytes, block: int = 1 << 22
) -> dict:
    """Spell the whole folded stream back out and compare it to the text.

    The one statement everything else rests on. ``verify_codec`` certifies the
    id-level codec; this certifies the *encoder*, which is where the alignment,
    the case decision and the fallback splice can all go wrong, and it does it
    on the real byte stream rather than on the argument that they cannot.

    Blocked so it can be run on the 230 MB train stream as well as on an eval
    suite: 53M tokens materialised as one Python list would cost a couple of
    gigabytes to check something the block loop checks for nothing.
    """
    stems = np.asarray(arrays["stem"], dtype=np.int64)
    n_roles = plan.vocab.n_roles
    roles = (
        np.asarray(arrays["role"], dtype=np.int64)
        if n_roles > 1
        else np.zeros_like(stems)
    )
    keys = stems * n_roles + roles
    uniq = np.unique(keys)
    table: dict[int, bytes] = {}
    for k in uniq.tolist():
        b = plan.vocab.spell(*divmod(k, n_roles))
        if b is None:
            s, r = divmod(k, n_roles)
            raise ValueError(
                f"the folded stream emits (stem {s}, role {r}), which has no "
                "spelling in this vocabulary"
            )
        table[k] = b
    lens = np.fromiter((len(table[k]) for k in uniq.tolist()), np.int64, uniq.size)
    per_token = lens[np.searchsorted(uniq, keys)]
    off = np.zeros(keys.size + 1, dtype=np.int64)
    np.cumsum(per_token, out=off[1:])
    if int(off[-1]) != len(raw):
        raise ValueError(
            f"the folded stream spells {int(off[-1])} bytes against a "
            f"{len(raw)}-byte text"
        )
    for a in range(0, keys.size, block):
        b = min(a + block, keys.size)
        got = b"".join(map(table.__getitem__, keys[a:b].tolist()))
        if got != raw[off[a] : off[b]]:
            bad = next(
                i
                for i in range(len(got))
                if got[i : i + 1] != raw[off[a] + i : off[a] + i + 1]
            )
            raise ValueError(
                "the folded stream does not round-trip: it diverges from the "
                f"text at byte {int(off[a]) + bad}"
            )
    return {"tokens": int(keys.size), "bytes": len(raw)}


# --- the plan -------------------------------------------------------------


@dataclass(frozen=True)
class ShrinkPlan:
    """A curriculum prefix, applied: the codec plus what each rung cost."""

    host: Vocabulary
    steps: tuple[Step, ...]
    codec: Rewrite
    train_tokens: int = 0
    host_tokens: int = 0
    stats: dict = field(default_factory=dict)

    @property
    def vocab(self) -> Vocabulary:
        return self.codec.dst

    def heads(self) -> dict[str, int]:
        """Head cardinalities for ``TransplantLM``.

        No role head before rule 1: an all-zero head would add parameters, a
        head-NLL term of exactly zero, and nothing else.
        """
        heads = {"stem": len(self.vocab)}
        if self.vocab.n_roles > 1:
            heads["role"] = self.vocab.n_roles
        return heads

    def apply_stream(self, host_ids: np.ndarray) -> dict[str, np.ndarray]:
        """Rewrite a stream of host ids into this plan's heads."""
        stems, roles = self.codec.apply(host_ids)
        out = {"stem": stems.astype(np.int64)}
        if self.vocab.n_roles > 1:
            out["role"] = roles.astype(np.int64)
        return out

    @property
    def fold_rung(self) -> int:
        """Index of the ``case_fold`` rung, or -1 if this plan has none."""
        for k, s in enumerate(self.steps):
            if s.case_invariant:
                return k
        return -1

    def _split_at_fold(self) -> tuple[Rewrite, Rewrite | None]:
        """The codec either side of the fold rung: (host -> pair, pair -> pair)."""
        k = self.fold_rung
        pre = self.steps[0].rewrite
        for s in self.steps[1 : k + 1]:
            pre = pre.then(s.rewrite)
        post = None
        for s in self.steps[k + 1 :]:
            post = s.rewrite if post is None else post.then(s.rewrite)
        return pre, post

    def fold_stream(self, enc: FoldEncoding) -> dict[str, np.ndarray]:
        """Rewrite a folded encoding into this plan's heads.

        The codec is split at the fold rung rather than applied whole. The case
        the encoder read from the text is a fact about the *merged* pair, so it
        has to be imposed there; anything after the fold (re-piecing) then
        rewrites the already-cased pair, which is how an uppercase word gets its
        pieces spelled in uppercase instead of being re-cased piece by piece.
        """
        k = self.fold_rung
        if k < 0:
            raise ValueError(
                "this plan has no case_fold rung, so a folded encoding has "
                "nowhere to put the case it read off the text"
            )
        pre, _ = self._split_at_fold()
        ids = np.asarray(enc.ids, dtype=np.int64)
        stems, roles = pre.apply(ids)
        if stems.size != ids.size:
            raise ValueError("the rungs up to the fold must be a relabeling")
        roles = roles.astype(np.int64)
        case = np.asarray(enc.case, dtype=np.int64)
        roles = np.where(case < 0, roles, (roles // 3) * 3 + case)
        for s in self.steps[k + 1 :]:
            stems, roles = s.rewrite.apply(stems, roles)
        out = {"stem": np.asarray(stems, dtype=np.int64)}
        if self.vocab.n_roles > 1:
            out["role"] = np.asarray(roles, dtype=np.int64)
        return out

    def source_rows(self, host_counts: np.ndarray) -> dict[str, np.ndarray]:
        """Which host rows feed each new row, and with what weight.

        Every new row is the frequency-weighted mean of the host rows that spell
        through it, which is the only rule that needs no special case: an
        unmerged row has exactly one source at weight one and comes out
        bit-identical, a merged row is centred on the mixture it will actually
        serve, and a stem introduced by re-piecing is centred on the words that
        re-piece through it. Weights are ``1 + count`` so a row that is active
        only through a perturbation suite still contributes.
        """
        n_src = self.codec.n_src_pairs // self.codec.src_roles
        src = np.arange(n_src, dtype=np.int64)
        length = self.codec.offsets[1:] - self.codec.offsets[:-1]
        weights = 1.0 + np.asarray(host_counts, dtype=np.float64)[:n_src]
        dst = self.codec.stems.astype(np.int64)
        out_src = np.repeat(src, length)
        out_w = np.repeat(weights, length)

        # The codec spells host ids in the case they already carry, which after a
        # fold is not everything the table has to cover: re-piecing an uppercase
        # word can use a stem no lowercase segmentation does. Measured on
        # ``shrink2-fold100``, exactly one row (the byte b'\xd0') came out with
        # no source, and ``shrink_init`` refuses to init a table with a row
        # nothing reaches. The variants are appended only where they name a
        # (row, host id) edge the codec did not already, and at weight 1 — the
        # same weight an active-but-unobserved row gets — so nothing that
        # already had a source is re-weighted and rule 0's bit-identical rows
        # stay single-source.
        extra = self._fold_source_edges()
        if extra is not None:
            edst, esrc = extra
            new = ~np.isin(edst * n_src + esrc, dst * n_src + out_src)
            dst = np.concatenate([dst, edst[new]])
            out_src = np.concatenate([out_src, esrc[new]])
            out_w = np.concatenate([out_w, np.ones(int(new.sum()))])
        return {"dst": dst, "src": out_src, "weight": out_w}

    def _fold_source_edges(self) -> tuple[np.ndarray, np.ndarray] | None:
        """(row, host id) edges reached by re-casing, or None without a fold."""
        if self.fold_rung < 0:
            return None
        pre, post = self._split_at_fold()
        folded = self.steps[self.fold_rung].rewrite.dst
        length = pre.offsets[1:] - pre.offsets[:-1]
        ids = np.flatnonzero(length == 1)
        stem = pre.stems[pre.offsets[ids]].astype(np.int64)
        base = (pre.roles[pre.offsets[ids]].astype(np.int64) // 3) * 3
        s, r, i = [], [], []
        for c in range(3):
            role = base + c
            legal = np.fromiter(
                (
                    folded.spell(int(a), int(b)) is not None
                    for a, b in zip(stem.tolist(), role.tolist(), strict=True)
                ),
                dtype=bool,
                count=ids.size,
            )
            s.append(stem[legal])
            r.append(role[legal])
            i.append(ids[legal])
        stem, role, ids = np.concatenate(s), np.concatenate(r), np.concatenate(i)
        if post is None:
            return stem, ids
        pieces, _, length = post._expand(stem * post.src_roles + role)
        return pieces.astype(np.int64), np.repeat(ids, length)

    def accounting(self, d_model: int) -> dict:
        """Embedding floats, derived from the mapping rather than typed in."""
        stem = len(self.vocab) * d_model
        # The role head is untied (``TransplantLM`` ties only the stem side), so
        # it is charged twice — a few thousand floats against a hundred million,
        # but charging it is what keeps the curve's y-axis honest.
        role = 2 * self.vocab.n_roles * d_model if self.vocab.n_roles > 1 else 0
        host = len(self.host) * d_model
        return {
            "n_stems": len(self.vocab),
            "n_roles": self.vocab.n_roles,
            "stem_floats": stem,
            "role_floats": role,
            "float_params": stem + role,
            "host_flat_floats": host,
            "compression_vs_host": host / (stem + role),
            "train_tokens": self.train_tokens,
            "token_inflation": (self.train_tokens / max(1, self.host_tokens)) - 1.0,
        }

    def summary(self) -> dict:
        return {
            "rules": [
                {
                    "rule": s.rule,
                    "params": s.params,
                    "n_stems": s.n_stems,
                    "n_roles": s.rewrite.dst.n_roles,
                    **s.stats,
                }
                for s in self.steps
            ],
            "n_stems": len(self.vocab),
            "n_roles": self.vocab.n_roles,
        }


def build(
    host: Vocabulary,
    active_ids: np.ndarray,
    host_stream: np.ndarray,
    rules: tuple[ShrinkRule, ...],
    verify: bool = True,
    fold_case: np.ndarray | None = None,
    host_tokens: int | None = None,
) -> ShrinkPlan:
    """Apply a curriculum prefix, re-measuring the stream between rungs.

    The stream is rewritten after every rule because rule 2's threshold is a
    *corpus* quantity: after rule 1 the counts of six case variants have become
    one stem's count, and after one re-piecing pass the dropped stems' mass has
    moved onto their pieces. Thresholding stale counts would make a progressive
    curriculum silently non-progressive.

    ``fold_case`` is ``FoldEncoding.case`` aligned to ``host_stream``, for a
    curriculum with a ``case_fold`` rung. It is imposed on the stream at that
    rung for the same reason the stream is rewritten at all: a later threshold
    reads counts, and after a fold the counts belong to the case-invariant
    stems, not to the ones the host's own tokenization would have produced.
    """
    vocab, codec = host, None
    stems = np.asarray(host_stream, dtype=np.int64)
    reach = np.asarray(active_ids, dtype=np.int64)
    roles = rroles = None
    steps: list[Step] = []
    for rule in rules:
        if rule.rule not in RULES:
            raise KeyError(f"unknown shrink rule {rule.rule!r}, have {sorted(RULES)}")
        step = RULES[rule.rule](vocab, rule, Context(host, stems, roles, reach, rroles))
        if verify:
            _verify(vocab, step)
        codec = step.rewrite if codec is None else codec.then(step.rewrite)
        stems, roles = step.rewrite.apply(stems, roles)
        reach, rroles = step.rewrite.apply(reach, rroles)
        if step.case_invariant:
            reach, rroles = _case_variants(reach, rroles, step.rewrite.dst)
            if fold_case is not None:
                case = np.asarray(fold_case, dtype=np.int64)
                have = np.asarray(roles, dtype=np.int64)
                roles = np.where(case < 0, have, (have // 3) * 3 + case)
        if step.rewrite.dst.n_roles == 1:
            roles = rroles = None
        vocab = step.rewrite.dst
        steps.append(step)
    if codec is None:
        raise ValueError("an empty curriculum shrinks nothing")
    return ShrinkPlan(
        host,
        tuple(steps),
        codec,
        int(stems.size),
        int(host_stream.size if host_tokens is None else host_tokens),
    )


def verify_codec(plan: ShrinkPlan, active_ids: np.ndarray) -> dict:
    """Spell every active host id through the *composed* codec.

    ``build`` checks each rung; this checks what they compose into, on the real
    vocabulary rather than on a fixture. It is the statement BPB comparability
    rests on: the curve's arms encode the same byte stream, so their
    denominators are identical and their numerators are commensurable.
    """
    pieces = 0
    for i in np.asarray(active_ids, dtype=np.int64).tolist():
        image = plan.codec.pairs(i)
        got = b"".join(plan.vocab.spell(s, r) or b"\xff\xfe" for s, r in image)
        if got != plan.host.stems[i]:
            raise ValueError(
                f"codec does not round-trip host id {i}: "
                f"{plan.host.stems[i]!r} -> {got!r}"
            )
        pieces += len(image)
    n = int(np.asarray(active_ids).size)
    return {"checked": n, "pieces_per_active_id": pieces / max(1, n)}


def _verify(src: Vocabulary, step: Step) -> None:
    """Every rewritten pair still spells the same bytes. The whole contract."""
    dst = step.rewrite.dst
    for key in range(step.rewrite.n_src_pairs):
        image = step.rewrite.pairs(*divmod(key, src.n_roles))
        if not image:
            continue
        want = src.spell(*divmod(key, src.n_roles))
        got = b"".join(dst.spell(s, r) or b"\xff\xfe" for s, r in image)
        if got != want:
            raise ValueError(
                f"{step.rule} does not round-trip: source pair {key} spells "
                f"{want!r} but its image spells {got!r}"
            )


# --- the host vocabulary --------------------------------------------------


def _byte_alphabet() -> dict[str, int]:
    """The GPT-2 byte-level alphabet, inverted. Reimplemented rather than
    imported so this module stays testable without ``transformers``."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    out = {chr(b): b for b in bs}
    n = 0
    for b in range(256):
        if b not in bs:
            out[chr(256 + n)] = b
            n += 1
    return out


def host_vocabulary(arm, table_size: int) -> Vocabulary:
    """The host's table as byte surfaces, padded to the model's row count.

    ``table_size`` is 151,936 for Qwen3 while the tokenizer has 151,669 entries;
    the padding rows get an empty surface and rule 0 deletes them, which is the
    same thing the active-vocab mask did to them.
    """
    tok = arm.tok
    u2b = _byte_alphabet()
    added = set(tok.get_added_vocab().values())
    n = min(len(tok), table_size)
    stems: list[bytes] = []
    for i, piece in enumerate(tok.convert_ids_to_tokens(list(range(n)))):
        if i in added:
            # A special token is atomic and has no byte-level spelling. None are
            # active on this corpus (measured, W0), so rule 0 deletes them; the
            # literal text keeps ``spell`` total in case that ever changes.
            stems.append(piece.encode("utf-8"))
            continue
        stems.append(bytes(u2b[c] for c in piece))
    stems += [b""] * (table_size - n)
    return Vocabulary(tuple(stems), n_roles=1)


# --- report ---------------------------------------------------------------


def main() -> None:
    import argparse
    import json
    from pathlib import Path
    import time

    from experiments.vocab_factorization.configs import (
        HOST_ARM,
        MASKED_NATIVE_BPB,
        SHRINK_CURRICULA,
        TransplantTrainSpec,
        active_cache_path,
        token_cache_path,
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    ap.add_argument("--arms-dir", type=Path, default=Path("data/arms"))
    ap.add_argument("--tokens-dir", type=Path, default=Path("data/tokens"))
    ap.add_argument("--out", type=Path, default=None, help="also write the JSON")
    ap.add_argument("--d-model", type=int, default=2048)
    ap.add_argument("--table-size", type=int, default=151936)
    ap.add_argument("--curriculum", default="default", choices=sorted(SHRINK_CURRICULA))
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument(
        "--no-verify-fold",
        action="store_true",
        help="skip spelling the whole 230 MB train stream back out (~40 s/rung)",
    )
    args = ap.parse_args()

    from experiments.vocab_factorization.build_arms import get_arm

    arm = get_arm(HOST_ARM, args.arms_dir, args.corpus)
    host = host_vocabulary(arm, args.table_size)
    curriculum = SHRINK_CURRICULA[args.curriculum]
    train_bytes = TransplantTrainSpec().train_bytes
    host_stream = np.load(
        token_cache_path(args.tokens_dir, HOST_ARM, "train", train_bytes)
    )["stem"]
    case = enc = raw = None
    if any(r.rule == "case_fold" for r in curriculum):
        from experiments.vocab_factorization import active_vocab
        from experiments.vocab_factorization.prepare import build_text, get_fold_tokens

        enc = get_fold_tokens(
            HOST_ARM,
            "train",
            train_bytes,
            args.corpus,
            args.arms_dir,
            args.tokens_dir,
            args.table_size,
            args.workers,
        )
        active = active_vocab.load_or_build_fold(
            HOST_ARM,
            args.table_size,
            args.corpus,
            args.arms_dir,
            args.tokens_dir,
            train_bytes,
            args.workers,
        )
        stream, case = enc.ids.astype(np.int64), enc.case
        print(f"fold: {json.dumps(enc.stats)}")
        if not args.no_verify_fold:
            # The corpus is round-tripped, not the argument that it would be.
            # Every rung below re-checks against these exact bytes.
            raw = build_text(args.corpus, "train", train_bytes).encode("utf-8")
    else:
        active = np.load(active_cache_path(args.tokens_dir, HOST_ARM))["ids"]
        stream = host_stream
    print(
        f"host table {len(host)}  active {active.size}  stream {stream.size / 1e6:.1f}M"
    )

    rows = []
    for k in range(1, len(curriculum) + 1):
        t0 = time.time()
        plan = build(
            host,
            active,
            stream,
            curriculum[:k],
            fold_case=case,
            host_tokens=int(host_stream.size),
        )
        acct = plan.accounting(args.d_model)
        check = verify_codec(plan, active)
        if raw is not None and plan.fold_rung >= 0:
            check |= verify_fold(plan, plan.fold_stream(enc), raw)
        last = plan.steps[-1]
        rows.append(
            {
                "stage": k,
                "rule": last.rule,
                "params": last.params,
                **acct,
                **check,
                "seconds": round(time.time() - t0, 1),
                "stats": last.stats,
            }
        )
        print(
            f"[{k}] {last.rule:14s} {str(last.params):24s} "
            f"V={acct['n_stems']:>6}  roles={acct['n_roles']}  "
            f"{acct['float_params'] / 1e6:7.2f}M floats "
            f"(x{acct['compression_vs_host']:.2f} vs host)  "
            f"tokens {plan.train_tokens / 1e6:.2f}M "
            f"({acct['token_inflation']:+.2%})  "
            f"{check['pieces_per_active_id']:.3f} pieces/id  "
            f"[{rows[-1]['seconds']:.0f}s]",
            flush=True,
        )
        for k2, v in last.stats.items():
            if k2 != "examples":
                print(f"       {k2}: {v}")
        arrow = "->" if last.rule == "repiece_tail" else "<-"
        for ex in last.stats.get("examples", [])[:6]:
            print(f"       {ex[0]!r} {arrow} {ex[1:]}")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        tag = "" if args.curriculum == "default" else f"_{args.curriculum}"
        path = args.out / f"shrink_curve{tag}.json"
        args.out.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "reference_bpb": MASKED_NATIVE_BPB,
                    "curriculum": args.curriculum,
                    "stages": rows,
                },
                indent=2,
            )
        )
        print(f"-> {path}")


if __name__ == "__main__":
    main()
