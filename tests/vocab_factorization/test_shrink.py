"""Progressive-shrink rules: the mapping, the accounting, and the round trip.

Every rule here deletes embedding rows, so the ways it can be wrong are the ways
a curve can look good for the wrong reason. Four of them are covered:

* a mapping that is not onto the smaller vocabulary leaves rows nothing can
  reach — parameters counted, never trained, and free BPB from a normalizer that
  is smaller than the table says;
* a mapping that loses a token makes the model stop predicting some of the text,
  which lowers BPB by *not modelling* the bytes rather than by modelling them
  better;
* accounting that disagrees with the mapping moves the x-axis without moving
  anything real;
* a second rule applied after the first has to compose — the whole track is the
  composition, and rule 2's thresholds are measured on the stream rule 1 left.

A tiny hand-built vocabulary stands in for Qwen3's 152k rows: the rules are pure
functions of byte surfaces, so nothing here needs a tokenizer or a GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.vocab_factorization import shrink
from experiments.vocab_factorization.configs import (
    SHRINK_CURRICULUM,
    SHRINK_FOLD_CURRICULUM,
    ShrinkRule,
)

# A vocabulary with all four cases the rules distinguish: case/space families,
# a mixed-case token that no role can express, a byte fragment that is not valid
# UTF-8, and the single bytes a re-piecing has to fall back on.
SURFACES: list[bytes] = [
    b"the",
    b" the",
    b"The",
    b" The",
    b"THE",
    b"cat",
    b" cat",
    b"Cat",
    b"aBc",  # mixed case: not one role
    b"\xd0",  # a UTF-8 lead byte on its own: no case at all
    b" ",
    b"!",
]
SURFACES += [bytes([b]) for b in range(256) if bytes([b]) not in SURFACES]

HOST = shrink.Vocabulary(tuple(SURFACES), n_roles=1)
# Control characters are inactive, as they are on the real corpus — which is
# what makes the byte floor a live concern rather than a hypothetical one.
INACTIVE = {i for i, s in enumerate(SURFACES) if len(s) == 1 and s[0] < 0x20}
ACTIVE = np.array(
    [i for i in range(len(SURFACES)) if i not in INACTIVE], dtype=np.int64
)


def _stream(ids: list[int], reps: int = 1) -> np.ndarray:
    return np.array(ids * reps, dtype=np.int64)


def _plan(rules, stream=None, active=ACTIVE):
    return shrink.build(
        HOST, active, _stream(active.tolist(), 40) if stream is None else stream, rules
    )


# The real curriculum's thresholds (100 / 1000) are calibrated to a 53M-token
# stream; on a toy one they would both bite at once and the two rungs would be
# indistinguishable. Same rules in the same order, thresholds scaled.
TOY_CURRICULUM = SHRINK_CURRICULUM[:2] + (
    ShrinkRule("repiece_tail", min_count=5),
    ShrinkRule("repiece_tail", min_count=40),
)
TOY_STREAM = np.concatenate([_stream(ACTIVE.tolist(), 3), _stream([0, 1, 2, 5, 6], 60)])


# --- the role algebra -----------------------------------------------------


def test_a_role_must_invert_or_it_is_not_a_role():
    """'ß'.upper() is 'SS', which does not lowercase back to 'ß'. A merge that
    accepted it would change the text it claims to only re-spell."""
    assert shrink._cased("ß".encode(), 2) is None
    assert shrink._cased(b"the", 2) == b"THE"
    # A form equal to the stem is a duplicate spelling, not a second role.
    assert shrink._cased(b"42", 1) is None


def test_canonicalisation_covers_exactly_what_it_can_undo():
    assert shrink._canonicalize(b" The") == (b"the", 4)  # space + title
    assert shrink._canonicalize(b"THE") == (b"the", 2)
    assert shrink._canonicalize(b"aBc") is None  # mixed case has no role
    assert shrink._canonicalize(b"\xd0") is None  # not valid UTF-8
    # A stem that begins with a space would make " x" spellable two ways.
    assert shrink._canonicalize(b"  x") is None


# --- each rule, on its own ------------------------------------------------


def test_rule_0_is_the_mask_expressed_as_deletion():
    active = np.array([0, 1, 5], dtype=np.int64)
    plan = _plan((ShrinkRule("mask"),), active=active)
    assert len(plan.vocab) == 3
    assert plan.vocab.stems == (b"the", b" the", b"cat")
    assert plan.heads() == {"stem": 3}  # no role head before rule 1
    for new, old in enumerate(active):
        assert plan.codec.pairs(int(old)) == [(new, 0)]


def test_rule_1_merges_case_and_space_into_one_row_plus_a_role():
    plan = _plan(SHRINK_CURRICULUM[:2])
    stems = plan.vocab.stems
    assert b"the" in stems and b"The" not in stems and b" the" not in stems
    assert plan.heads() == {"stem": len(stems), "role": 6}
    # Five surfaces of one lexeme collapse onto one row, distinguished by role.
    the = stems.index(b"the")
    got = {
        SURFACES[i]: plan.codec.pairs(i)[0]
        for i in range(len(SURFACES))
        if SURFACES[i] in (b"the", b" the", b"The", b" The", b"THE")
    }
    assert {p[0] for p in got.values()} == {the}
    assert len(set(got.values())) == 5, "distinct surfaces must keep distinct roles"
    # What no role can express stays its own row.
    assert b"aBc" in stems and b"\xd0" in stems


def test_rule_2_spells_the_tail_out_of_what_survives():
    """A stem below the threshold stops being a row and becomes a segmentation."""
    stream = _stream([0, 1, 2, 5], 50)  # only 'the' family and 'cat' are frequent
    plan = shrink.build(HOST, ACTIVE, stream, SHRINK_CURRICULUM[:2])
    before = len(plan.vocab)
    tail = (ShrinkRule("repiece_tail", min_count=10),)
    plan2 = shrink.build(HOST, ACTIVE, stream, SHRINK_CURRICULUM[:2] + tail)
    assert len(plan2.vocab) < before
    assert b"the" in plan2.vocab.stems
    # 'aBc' never occurs in training but is active, so it is still spellable —
    # out of surviving pieces rather than out of its own row.
    pieces = plan2.codec.pairs(SURFACES.index(b"aBc"))
    assert len(pieces) > 1
    assert b"".join(plan2.vocab.spell(s, r) for s, r in pieces) == b"aBc"


def test_reachability_is_the_active_set_not_the_training_corpus():
    """2,480 of Qwen3's active ids never occur in training — they come from the
    perturbation suites. Thresholding reachability on the corpus would make them
    impossible to emit, which under a restricted softmax is unbounded BPB."""
    stream = _stream([0, 1, 2, 5], 50)
    plan = shrink.build(
        HOST,
        ACTIVE,
        stream,
        SHRINK_CURRICULUM[:2] + (ShrinkRule("repiece_tail", min_count=10),),
    )
    unseen = [i for i in ACTIVE.tolist() if i not in {0, 1, 2, 5}]
    out = plan.apply_stream(np.array(unseen, dtype=np.int64))
    assert out["stem"].size >= len(unseen)


# --- the invariants the curve rests on ------------------------------------


@pytest.mark.parametrize("k", range(1, 5))
def test_the_mapping_is_onto_the_smaller_vocabulary(k):
    """Every row of the shrunk table is reached by some active token. A row that
    is not is a parameter nobody trains and a softmax competitor nobody can
    emit — it would make the x-axis a fiction."""
    plan = shrink.build(HOST, ACTIVE, TOY_STREAM, TOY_CURRICULUM[:k])
    reached = set()
    for i in ACTIVE.tolist():
        reached.update(s for s, _ in plan.codec.pairs(i))
    assert reached == set(range(len(plan.vocab)))
    assert len(plan.vocab) <= len(ACTIVE)


@pytest.mark.parametrize("k", range(1, 5))
def test_no_active_token_becomes_unreachable(k):
    """Total on the active set, and every token still spells its own bytes:
    the stream may lengthen, the byte string it encodes may not change."""
    plan = shrink.build(HOST, ACTIVE, TOY_STREAM, TOY_CURRICULUM[:k])
    for i in ACTIVE.tolist():
        pairs = plan.codec.pairs(i)
        assert pairs, f"host id {i} ({SURFACES[i]!r}) maps to nothing"
        spelled = b"".join(plan.vocab.spell(s, r) for s, r in pairs)
        assert spelled == HOST.stems[i]


def test_an_inactive_id_is_refused_rather_than_dropped():
    """A dropped token would silently shorten the stream and lower BPB by not
    modelling those bytes at all."""
    active = np.array([0, 1, 5], dtype=np.int64)
    plan = _plan((ShrinkRule("mask"),), active=active)
    with pytest.raises(ValueError, match="maps to nothing"):
        plan.apply_stream(np.array([0, 7], dtype=np.int64))


def test_repiecing_lengthens_the_stream_and_says_so():
    stream = _stream([0, 1, 2, 5], 50)
    plan = shrink.build(
        HOST,
        ACTIVE,
        stream,
        SHRINK_CURRICULUM[:2] + (ShrinkRule("repiece_tail", min_count=10),),
    )
    long_stream = _stream(ACTIVE.tolist(), 5)
    out = plan.apply_stream(long_stream)
    assert out["stem"].size > long_stream.size
    assert plan.steps[-1].stats["dropped"] > 0


def test_accounting_matches_the_mapping():
    plan = _plan(SHRINK_CURRICULUM[:2])
    acct = plan.accounting(d_model=16)
    assert acct["n_stems"] == len(plan.vocab.stems)
    assert acct["stem_floats"] == len(plan.vocab) * 16
    # The role head is untied, so it is charged on both sides.
    assert acct["role_floats"] == 2 * 6 * 16
    assert acct["float_params"] == acct["stem_floats"] + acct["role_floats"]
    assert acct["host_flat_floats"] == len(SURFACES) * 16
    assert acct["compression_vs_host"] == pytest.approx(
        acct["host_flat_floats"] / acct["float_params"]
    )


def test_a_relabeling_does_not_change_the_token_count():
    """Rules 0 and 1 emit exactly one pair per token, so the curve's first two
    points differ in vocabulary size and in nothing else."""
    stream = _stream(ACTIVE.tolist(), 30)
    plan = shrink.build(HOST, ACTIVE, stream, SHRINK_CURRICULUM[:2])
    out = plan.apply_stream(stream)
    assert out["stem"].shape == stream.shape
    assert out["role"].shape == stream.shape
    assert plan.accounting(16)["token_inflation"] == 0.0


# --- composition ----------------------------------------------------------


def test_the_curriculum_composes_and_only_shrinks():
    sizes = [
        len(shrink.build(HOST, ACTIVE, TOY_STREAM, TOY_CURRICULUM[:k]).vocab)
        for k in range(1, len(TOY_CURRICULUM) + 1)
    ]
    assert sizes == sorted(sizes, reverse=True), sizes
    assert sizes[0] == len(ACTIVE)
    # The byte floor is the hard lower bound: a vocabulary that cannot spell
    # every byte cannot re-piece, so no rung goes below it.
    assert sizes[-1] >= 1


def test_rule_2_after_rule_1_thresholds_the_merged_counts():
    """The whole point of re-measuring between rungs: after rule 1 a lexeme's
    five variants are one stem with their summed count, so a threshold that
    would have dropped each variant keeps the merged row."""
    stream = _stream([1, 2, 3, 4], 5)  # 5 of each " the"/"The"/" The"/"THE"
    tail = (ShrinkRule("repiece_tail", min_count=12),)
    with_merge = shrink.build(HOST, ACTIVE, stream, SHRINK_CURRICULUM[:2] + tail)
    without = shrink.build(HOST, ACTIVE, stream, SHRINK_CURRICULUM[:1] + tail)
    assert b"the" in with_merge.vocab.stems
    assert b" The" not in without.vocab.stems  # 5 < 12 on its own


def test_composition_agrees_with_applying_the_steps_one_at_a_time():
    stream = _stream(ACTIVE.tolist(), 20)
    plan = shrink.build(HOST, ACTIVE, stream, TOY_CURRICULUM)
    stems, roles = np.asarray(stream), None
    for step in plan.steps:
        stems, roles = step.rewrite.apply(stems, roles)
    assert np.array_equal(plan.apply_stream(stream)["stem"], stems)
    assert np.array_equal(plan.apply_stream(stream)["role"], roles)


def test_a_broken_rule_is_caught_by_the_round_trip_check():
    """``build``'s verification is the contract that keeps BPB comparable, so it
    has to actually fire when a rule stops spelling the same bytes."""
    bad = shrink.Vocabulary((b"the", b"cat"), n_roles=1)
    step = shrink.Step(
        "broken",
        {},
        shrink._rewrite_from_lists(len(HOST), 1, bad, {0: [(1, 0)]}),
        {},
    )
    with pytest.raises(ValueError, match="does not round-trip"):
        shrink._verify(HOST, step)


# --- case folding at tokenization time ------------------------------------
#
# Rule 1 merges ids; the fold re-tokenizes. The difference only shows up on text
# whose casing changes the *segmentation*, so the toy tokenizer below is a real
# byte-level greedy segmenter over the same surfaces — " THE" is one piece and
# " ThE" is not, exactly as on the host.


class ToyTokenizer:
    """Longest-match byte segmentation over ``SURFACES``, shaped like HF's."""

    def __init__(self, surfaces: list[bytes]) -> None:
        self.index = {}
        for i, s in enumerate(surfaces):
            self.index.setdefault(s, i)
        self.longest = max(len(s) for s in surfaces)

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        b = text.encode("utf-8")
        ids: list[int] = []
        i = 0
        while i < len(b):
            for ln in range(min(self.longest, len(b) - i), 0, -1):
                hit = self.index.get(b[i : i + ln])
                if hit is not None:
                    ids.append(hit)
                    i += ln
                    break
            else:  # pragma: no cover - SURFACES contains every single byte
                raise AssertionError("the toy vocabulary cannot spell a byte")
        return {"input_ids": ids}


TOK = ToyTokenizer(SURFACES)


def _fold_plan(text: str, rules=SHRINK_FOLD_CURRICULUM[:2], extra: str = ""):
    """Build a fold plan whose active set is what folding ``text`` emits."""
    enc = shrink.fold_encode(text + extra, TOK, HOST)
    active = np.union1d(np.unique(enc.ids.astype(np.int64)), ACTIVE)
    plan = shrink.build(
        HOST,
        active,
        enc.ids.astype(np.int64),
        rules,
        fold_case=enc.case,
    )
    return plan, enc


def test_fold_makes_the_stem_stream_case_invariant():
    """The measurement this whole rule exists for. Rule 1 leaves ``upper`` and
    ``lower`` sharing 36.9% of their stem mass on the real corpus because the
    host fragments uppercase text first; a fold has to leave them identical."""
    plan, _ = _fold_plan(" the cat the cat")
    low = plan.fold_stream(shrink.fold_encode(" the cat", TOK, HOST))
    up = plan.fold_stream(shrink.fold_encode(" THE CAT", TOK, HOST))
    title = plan.fold_stream(shrink.fold_encode(" The Cat", TOK, HOST))
    assert np.array_equal(low["stem"], up["stem"])
    assert np.array_equal(low["stem"], title["stem"])
    assert not np.array_equal(low["role"], up["role"])
    # And the merge, on the same text, does not: " THE" is one host id but
    # " CAT" is not in this vocabulary, so it fragments before any merge.
    merge = shrink.build(
        HOST, ACTIVE, _stream(ACTIVE.tolist(), 40), SHRINK_CURRICULUM[:2]
    )
    assert not np.array_equal(
        merge.apply_stream(np.array(TOK(" the cat")["input_ids"]))["stem"],
        merge.apply_stream(np.array(TOK(" THE CAT")["input_ids"]))["stem"],
    )


@pytest.mark.parametrize(
    "text",
    [
        " the cat",
        " THE CAT",
        " The Cat",
        " tHe CaT!",  # mixed case: the fallback path
        "aBc \xd0\x9f",  # a surface no role expresses, next to a byte fragment
        "İstanbul",  # 'İ'.lower() is two characters: the lowering expands
        " stra\xdfe STRASSE",  # 'ß'.upper() is 'SS', which does not invert
        "  the   THE  ",  # multi-space surfaces stay verbatim
        "",
    ],
)
def test_the_fold_round_trips_the_text_it_encodes(text):
    """Losslessness is checked by spelling the stream back out, never argued."""
    plan, _ = _fold_plan(" the cat", extra=text)
    arrays = plan.fold_stream(shrink.fold_encode(text, TOK, HOST))
    assert shrink.verify_fold(plan, arrays, text.encode("utf-8"))["bytes"] == len(
        text.encode("utf-8")
    )


def test_mixed_case_falls_back_to_the_original_tokenization_and_says_so():
    """A token that spans a case change is not expressible as one role, so the
    region is re-tokenized from the *original* text. The rate is reported
    because this is the one place the fold gives up its own guarantee: on the
    real corpus it is 0.18% of clean tokens and 45% of ``randcase`` ones."""
    plan, _ = _fold_plan(" the cat thE")
    enc = shrink.fold_encode(" thE", TOK, HOST)
    assert enc.stats["fallback_groups"] == 1
    assert enc.stats["fallback_tokens"] > 1, "the region re-fragments"
    assert shrink.verify_fold(plan, plan.fold_stream(enc), b" thE")


def test_mixed_case_split_across_tokens_needs_no_fallback():
    """Which is why ``randcase`` is not a total loss: where the tokenizer
    already cuts at the case change, each piece has a case of its own and the
    fold expresses it as a role instead of giving up."""
    enc = shrink.fold_encode("aBc", TOK, HOST)  # 'abc' is not one piece here
    assert enc.stats["fallback_tokens"] == 0
    assert enc.stats["recased_tokens"] == 1


def test_the_fallback_adds_no_row_the_merge_would_not_have():
    """The cost of the fallback is fragmentation, not vocabulary. Study 1's
    verbatim path added stems; here the fallback emits *host ids*, and a surface
    no role expresses is already a verbatim row under rule 1. On the real
    vocabulary that shows up as 503 of the fold's 45,742 rows still carrying an
    uppercase letter, against 4,395 of the merge's 48,703."""
    stream = _stream(ACTIVE.tolist(), 40)
    merge = shrink.build(HOST, ACTIVE, stream, SHRINK_CURRICULUM[:2])
    fold, _ = _fold_plan(" the cat thE")
    assert b"aBc" in merge.vocab.stems and b"aBc" in fold.vocab.stems
    assert set(fold.vocab.stems) <= set(merge.vocab.stems)


def test_a_clean_lowercase_text_never_takes_the_fallback():
    enc = shrink.fold_encode(" the cat the cat!", TOK, HOST)
    assert enc.stats["fallback_tokens"] == 0
    assert enc.stats["recased_tokens"] == 0
    assert enc.stats["unattributable_boundaries"] == 0


def test_min_group_1_costs_no_rows():
    """The fold forces ``min_group=1`` so the encoder's decision is a function of
    a surface. That is free: two surfaces sharing a lowercase core would be a
    group of two, so a singleton's core collides with nothing."""
    stream = _stream(ACTIVE.tolist(), 40)
    one = shrink.build(
        HOST, ACTIVE, stream, SHRINK_CURRICULUM[:1] + (ShrinkRule("case_space", 0, 1),)
    )
    two = shrink.build(HOST, ACTIVE, stream, SHRINK_CURRICULUM[:2])
    assert len(one.vocab) == len(two.vocab)


def test_case_fold_refuses_a_group_threshold_it_cannot_honour():
    with pytest.raises(ValueError, match="min_group=1"):
        shrink.build(
            HOST,
            ACTIVE,
            _stream(ACTIVE.tolist(), 4),
            SHRINK_CURRICULUM[:1] + (ShrinkRule("case_fold", 0, 2),),
        )


def test_a_fold_makes_every_case_form_reachable_for_the_re_piecing():
    """After a fold, "which casings occur" is no longer a fact about the corpus:
    the arm is case-invariant by construction, so rule 2 has to be able to spell
    an uppercase word the training stream never contained."""
    plan, _ = _fold_plan(" the cat " * 30, rules=SHRINK_FOLD_CURRICULUM[:3])
    arrays = plan.fold_stream(shrink.fold_encode(" THE CAT", TOK, HOST))
    assert shrink.verify_fold(plan, arrays, b" THE CAT")


def test_a_repiecing_after_a_fold_carries_the_case_onto_the_pieces():
    """The case is imposed at the fold rung, not on the final pairs, so a word
    that re-pieces gets *every* piece spelled in the case it was read in."""
    plan, _ = _fold_plan(
        " the the the cat",
        rules=SHRINK_FOLD_CURRICULUM[:2] + (ShrinkRule("repiece_tail", min_count=2),),
    )
    arrays = plan.fold_stream(shrink.fold_encode(" CAT", TOK, HOST))
    assert arrays["stem"].size > 1, "the tail stem should have been re-pieced"
    assert shrink.verify_fold(plan, arrays, b" CAT")


def test_every_row_of_a_folded_table_still_has_a_host_source():
    """``shrink_init`` refuses a table with a row nothing reaches, and after a
    fold the codec alone no longer covers one: re-piecing an *uppercase* word
    can use a stem no lowercase segmentation does. Measured on the real
    vocabulary, that was one row of ``shrink2-fold100``'s 16,632."""
    rules = SHRINK_FOLD_CURRICULUM[:2] + (ShrinkRule("repiece_tail", min_count=3),)
    plan, enc = _fold_plan(" the cat " * 20, rules=rules)
    counts = np.bincount(enc.ids.astype(np.int64), minlength=len(SURFACES))
    rows = plan.source_rows(counts)
    assert set(rows["dst"].tolist()) == set(range(len(plan.vocab)))


def test_the_fold_leaves_a_single_source_row_single_source():
    """The extra edges carry weight 1 and are added only where the codec named
    no edge at all, so rule 0's rows still come out bit-identical — which is
    what makes a mask rung reproduce the masked native reference exactly."""
    plan, enc = _fold_plan(" the cat")
    counts = np.bincount(enc.ids.astype(np.int64), minlength=len(SURFACES))
    rows = plan.source_rows(counts)
    seen = list(zip(rows["dst"].tolist(), rows["src"].tolist()))
    assert len(seen) == len(set(seen)), "an edge was counted twice"
    fan = np.bincount(rows["dst"], minlength=len(plan.vocab))
    verbatim = plan.vocab.stems.index(b"aBc")  # no role, so exactly one source
    assert fan[verbatim] == 1


def test_the_fold_needs_a_fold_rung_to_put_the_case_in():
    plan = _plan(SHRINK_CURRICULUM[:2])
    with pytest.raises(ValueError, match="no case_fold rung"):
        plan.fold_stream(shrink.fold_encode(" the", TOK, HOST))


# --- asking the codec before applying it ----------------------------------
#
# On the corpus, every text the BPB harness scores is inside the active set by
# construction, so ``apply`` raising is the right and only behaviour. A
# downstream benchmark is the case that convention does not cover: HellaSwag
# emits 2,072 of 3,166,583 host tokens (207 surfaces — " mascara",
# " tablespoons") this ru+en corpus never does.


def test_covers_names_exactly_the_ids_apply_would_accept():
    active = np.array([0, 1, 5], dtype=np.int64)
    plan = _plan((ShrinkRule("mask"),), active=active)
    assert plan.covers(np.array([0, 7, 5, 9], dtype=np.int64)).tolist() == [
        True,
        False,
        True,
        False,
    ]
    # The two must agree, or asking first would be a different question from
    # the one applying answers.
    with pytest.raises(ValueError, match="maps to nothing"):
        plan.apply_stream(np.array([7], dtype=np.int64))


def test_the_segmenter_spells_out_of_corpus_text_out_of_what_remains():
    """Rule 2's operation, pointed at a surface the corpus never contained."""
    active = np.array(
        [i for i in ACTIVE.tolist() if SURFACES[i] != b" cat"], dtype=np.int64
    )
    plan = _plan((ShrinkRule("mask"),), active=active)
    assert not plan.covers(np.array([SURFACES.index(b" cat")]))[0]
    pieces = shrink.Segmenter(plan.vocab)(b" cat")
    assert pieces is not None and len(pieces) > 1
    assert b"".join(plan.vocab.spell(s, r) for s, r in pieces) == b" cat"


def test_the_segmenter_admits_when_the_byte_floor_is_incomplete():
    """Rule 0 deletes 66 of the 256 single-byte host tokens on the real corpus,
    so before a ``repiece_tail`` rung a surface can have no segmentation at all.
    Returning None is what lets a caller stop rather than drop the bytes."""
    only_the = np.array([SURFACES.index(b"the")], dtype=np.int64)
    plan = _plan((ShrinkRule("mask"),), active=only_the)
    assert shrink.Segmenter(plan.vocab)(b"cat") is None


@pytest.mark.parametrize(
    "text", [" The", " THE", "the", " the", "Cat", " tHe", "aBc", "İstanbul"]
)
def test_fold_surface_inverts_the_case_the_encoder_read(text):
    """``fold_encode`` emits (lowercase id, case); this puts the case back. The
    property is that concatenating them spells the text again — which is what
    makes it safe to re-piece a folded token the codec has no row for."""
    enc = shrink.fold_encode(text, TOK, HOST)
    got = b"".join(
        shrink.fold_surface(HOST, int(i), int(c))
        for i, c in zip(enc.ids.tolist(), enc.case.tolist(), strict=True)
    )
    assert got == text.encode("utf-8")


# --- the init the healing train starts from -------------------------------


def test_source_rows_cover_every_new_row_with_the_right_weights():
    stream = _stream([0, 1, 2], 10)
    plan = shrink.build(HOST, ACTIVE, stream, SHRINK_CURRICULUM[:2])
    counts = np.bincount(stream, minlength=len(SURFACES))
    rows = plan.source_rows(counts)
    assert rows["dst"].size == rows["src"].size == rows["weight"].size
    assert set(rows["dst"].tolist()) == set(range(len(plan.vocab)))
    # Weight is 1 + count, so a row active only through a perturbation suite
    # still contributes rather than dividing by zero.
    assert rows["weight"].min() == 1.0
    the = plan.vocab.stems.index(b"the")
    assert rows["weight"][rows["dst"] == the].max() == 11.0


def test_the_init_leaves_an_unmerged_row_bit_identical():
    """Rule 0 is a permutation of the host's own rows, which is what makes the
    ``shrink0-mask`` rung reproduce the masked native reference exactly rather
    than nearly."""
    torch = pytest.importorskip("torch")
    from experiments.vocab_factorization import transplant
    from tests.vocab_factorization.test_w0 import StubBody

    active = np.array([0, 5, 9], dtype=np.int64)
    plan = _plan((ShrinkRule("mask"),), active=active)
    host_emb = torch.randn(len(SURFACES), 8)
    model = transplant.TransplantLM(StubBody(8), plan.heads(), 8)
    stats = transplant.shrink_init(
        model,
        plan.source_rows(np.zeros(len(SURFACES))),
        host_emb,
        verbose=False,
    )
    assert stats["merged_rows"] == 0
    assert torch.allclose(model.emb["stem"].weight, host_emb[active], atol=1e-6)
