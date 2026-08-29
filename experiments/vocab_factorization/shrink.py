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

Measured on Qwen3-1.7B's vocabulary and the real 230 MB stream (run the module,
2026-08-30; 24 s of CPU for the deepest rung, so nothing here needs a cache):

=====================================  ======  ========  ====================
rule 0  active-vocab mask               74,957  153.51M   W0's mask, as deletion
rule 1  case/space -> stem + role bit   48,703   99.77M   token count unchanged
rule 2  re-piece the tail (< 100)       17,385   35.63M   +1.81% tokens
rule 2  re-piece the tail (< 1000)       4,783    9.82M   +14.07% tokens
=====================================  ======  ========  ====================

Run::

    PYTHONPATH=. python -m experiments.vocab_factorization.shrink
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
    """One curriculum rung: the rule, its rewrite, and what it actually did."""

    rule: str
    params: dict
    rewrite: Rewrite
    stats: dict

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
    "repiece_tail": _rule_repiece_tail,
}


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
        return {
            "dst": self.codec.stems.astype(np.int64),
            "src": np.repeat(src, length),
            "weight": np.repeat(weights, length),
        }

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
) -> ShrinkPlan:
    """Apply a curriculum prefix, re-measuring the stream between rungs.

    The stream is rewritten after every rule because rule 2's threshold is a
    *corpus* quantity: after rule 1 the counts of six case variants have become
    one stem's count, and after one re-piecing pass the dropped stems' mass has
    moved onto their pieces. Thresholding stale counts would make a progressive
    curriculum silently non-progressive.
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
        if step.rewrite.dst.n_roles == 1:
            roles = rroles = None
        vocab = step.rewrite.dst
        steps.append(step)
    if codec is None:
        raise ValueError("an empty curriculum shrinks nothing")
    return ShrinkPlan(host, tuple(steps), codec, int(stems.size), int(host_stream.size))


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
        SHRINK_CURRICULUM,
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
    args = ap.parse_args()

    from experiments.vocab_factorization.build_arms import get_arm

    arm = get_arm(HOST_ARM, args.arms_dir, args.corpus)
    host = host_vocabulary(arm, args.table_size)
    active = np.load(active_cache_path(args.tokens_dir, HOST_ARM))["ids"]
    stream = np.load(
        token_cache_path(
            args.tokens_dir, HOST_ARM, "train", TransplantTrainSpec().train_bytes
        )
    )["stem"]
    print(
        f"host table {len(host)}  active {active.size}  stream {stream.size / 1e6:.1f}M"
    )

    rows = []
    for k in range(1, len(SHRINK_CURRICULUM) + 1):
        t0 = time.time()
        plan = build(host, active, stream, SHRINK_CURRICULUM[:k])
        acct = plan.accounting(args.d_model)
        check = verify_codec(plan, active)
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
        arrow = "<-" if last.rule == "case_space" else "->"
        for ex in last.stats.get("examples", [])[:6]:
            print(f"       {ex[0]!r} {arrow} {ex[1:]}")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "shrink_curve.json").write_text(
            json.dumps({"reference_bpb": MASKED_NATIVE_BPB, "stages": rows}, indent=2)
        )
        print(f"-> {args.out / 'shrink_curve.json'}")


if __name__ == "__main__":
    main()
