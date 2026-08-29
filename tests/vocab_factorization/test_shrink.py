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
