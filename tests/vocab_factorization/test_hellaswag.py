"""HellaSwag scoring arithmetic, on CPU with a stub body.

The failure modes worth guarding are all silent ones. An off-by-one in the
continuation boundary scores part of the context and still returns a plausible
accuracy. Padding that is not fully discarded adds a small wrong number to every
short choice, which systematically favours long ones. Dropping the feature heads
lets a factorized arm hide its case cost in a side head. None of these crash;
they just produce a number that means something other than what it is called.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from experiments.vocab_factorization import hellaswag, shrink, transplant  # noqa: E402
from experiments.vocab_factorization.configs import (  # noqa: E402
    SHRINK_CURRICULUM,
    SHRINK_FOLD_CURRICULUM,
    ShrinkRule,
)
from experiments.vocab_factorization.tokenization import Encoding  # noqa: E402
from tests.vocab_factorization.test_shrink import (  # noqa: E402
    ACTIVE,
    HOST,
    SURFACES,
    TOK,
    _stream,
)
from tests.vocab_factorization.test_w0 import D_MODEL, VOCAB, StubBody  # noqa: E402


def _model(heads=None):
    torch.manual_seed(0)
    return transplant.TransplantLM(StubBody(), heads or {"stem": VOCAB}, D_MODEL)


def _seq(ids, ctx_len, cont_bytes=10, heads=("stem",), role=None):
    arrays = {"stem": np.asarray(ids, dtype=np.int64)}
    for h in heads:
        if h != "stem":
            arrays[h] = np.asarray(
                role if role is not None else [0] * len(ids), dtype=np.int64
            )
    return {"arrays": arrays, "ctx_len": ctx_len, "cont_bytes": cont_bytes}


# --- preprocessing --------------------------------------------------------


def test_preprocess_matches_the_canonical_cleanup():
    # The trailing space is the harness's, not an oversight: its preprocess
    # strips only at the front and never re-strips after removing markup.
    # Matching it exactly is what keeps these accuracies comparable to
    # published ones, so the fidelity is the assertion.
    assert hellaswag.preprocess("  Wash hands [title] Rinse  well [step] ") == (
        "Wash hands. Rinse well "
    )


# --- the continuation boundary -------------------------------------------


def test_keep_mask_covers_exactly_the_continuation():
    """Target position j predicts token j+1, so continuation starts at ctx_len-1."""
    seqs = [_seq([1, 2, 3, 4, 5, 6], ctx_len=3)]
    _, targets, keep = _pad_batch_cpu(seqs)
    # tokens 3,4,5 (0-based indices 3..5) are the continuation -> target j=2,3,4
    assert keep[0].tolist() == [False, False, True, True, True]
    assert targets["stem"][0].tolist() == [2, 3, 4, 5, 6]


def test_padding_contributes_nothing():
    """A short choice batched with a long one must score as if it were alone."""
    model = _model()
    short = _seq([1, 2, 3, 4], ctx_len=2)
    long = _seq([5, 6, 7, 8, 9, 10, 11], ctx_len=2)

    alone = hellaswag.score(
        model, [{"gold": 0, "seqs": [short] * 4}], {"stem": VOCAB}, "cpu", batch=4
    )[0, 0]
    together = hellaswag.score(
        model,
        [{"gold": 0, "seqs": [short, long, short, long]}],
        {"stem": VOCAB},
        "cpu",
        batch=4,
    )[0, 0]
    assert together == pytest.approx(alone, rel=1e-6)


def _pad_batch_cpu(seqs):
    return hellaswag._pad_batch(seqs, {"stem": VOCAB}, torch.device("cpu"))


# --- the score itself -----------------------------------------------------


def test_score_equals_the_summed_continuation_nll():
    """Hand-check against token_nll on one unpadded sequence."""
    model = _model()
    seq = _seq([3, 1, 4, 1, 5, 9], ctx_len=3)
    got = hellaswag.score(
        model, [{"gold": 0, "seqs": [seq] * 4}], {"stem": VOCAB}, "cpu", batch=1
    )[0, 0]

    ids = torch.tensor([[3, 1, 4, 1, 5, 9]])
    per_head = model.token_nll({"stem": ids[:, :-1]}, {"stem": ids[:, 1:]})
    expected = float(per_head["stem"][0, 2:].sum())  # ctx_len-1 = 2 onwards
    assert got == pytest.approx(expected, rel=1e-6)


def test_every_head_is_summed():
    """A factorized arm's side heads must not be free."""
    heads = {"stem": VOCAB, "role": 4}
    model = _model(heads)
    with torch.no_grad():  # make the role head cost something
        model.out["role"].weight.normal_()
        model.emb["role"].weight.normal_()
    ids = [1, 2, 3, 4, 5]
    role = [0, 1, 2, 3, 1]
    seq = _seq(ids, ctx_len=2, heads=("stem", "role"), role=role)
    total = hellaswag.score(
        model, [{"gold": 0, "seqs": [seq] * 4}], heads, "cpu", batch=2
    )[0, 0]

    t_ids, t_role = torch.tensor([ids]), torch.tensor([role])
    per_head = model.token_nll(
        {"stem": t_ids[:, :-1], "role": t_role[:, :-1]},
        {"stem": t_ids[:, 1:], "role": t_role[:, 1:]},
    )
    stem_part = float(per_head["stem"][0, 1:].sum())
    role_part = float(per_head["role"][0, 1:].sum())
    assert role_part > 0
    assert total == pytest.approx(stem_part + role_part, rel=1e-6)
    assert total > stem_part


# --- accuracy -------------------------------------------------------------


def test_acc_and_acc_norm_can_disagree():
    """acc favours short endings; the byte-normalised variant is why it exists."""
    encoded = [
        {
            "gold": 1,
            "seqs": [
                _seq([1, 2], 1, cont_bytes=2),  # cheap because it is tiny
                _seq([1, 2], 1, cont_bytes=40),  # gold: better per byte
                _seq([1, 2], 1, cont_bytes=40),
                _seq([1, 2], 1, cont_bytes=40),
            ],
        }
    ]
    nll = np.array([[10.0, 12.0, 30.0, 30.0]])
    res = hellaswag.accuracy(nll, encoded)
    assert res["acc"] == 0.0  # argmin picks choice 0
    assert res["acc_norm"] == 1.0  # 12/40 = 0.3 beats 10/2 = 5.0
    assert res["n"] == 1


def test_coverage_counts_only_continuation_tokens():
    active = np.array([1, 2, 3], dtype=np.int64)
    encoded = [{"gold": 0, "seqs": [_seq([9, 9, 1, 7], ctx_len=2)] * 4}]
    cov = hellaswag.coverage(encoded, active)
    # context ids 9,9 are ignored; of the continuation (1, 7) only 7 is inactive
    assert cov["continuation_tokens"] == 8
    assert cov["tokens_outside_active"] == 4
    assert cov["fraction_outside_active"] == pytest.approx(0.5)


# --- progressive-shrink arms ----------------------------------------------
#
# A shrink arm's model indexes a *rewritten* vocabulary, so everything between
# "here is a document" and "here are the arrays" is different, and every way of
# getting it wrong returns a number rather than an error. The toy vocabulary
# from ``test_shrink`` stands in for Qwen3's 152k rows: the codec is a pure
# function of byte surfaces, so none of this needs a tokenizer or a GPU.


class ToyArm:
    """The three things ``hellaswag`` asks an arm for: heads, encode, tok."""

    heads = {"stem": len(SURFACES)}
    tok = TOK

    def encode(self, text: str) -> Encoding:
        ids = TOK(text)["input_ids"]
        return Encoding(
            stem=np.asarray(ids, dtype=np.int32),
            role=np.zeros(len(ids), dtype=np.int8),
            suffix=np.zeros(len(ids), dtype=np.int16),
        )


ARM = ToyArm()

# " the"/"the" are frequent, everything else is rare, so a ``repiece_tail`` rung
# keeps the lexeme and re-pieces " cat" — which is what makes the continuation
# boundary move.
SKEWED = np.concatenate(
    [
        _stream(ACTIVE.tolist(), 2),
        _stream([SURFACES.index(b" the"), SURFACES.index(b"the")], 60),
    ]
)


def _doc(query: str, choices: list[str], gold: int = 0) -> dict:
    return {"query": query, "choices": choices, "gold": gold}


def _spell(plan, arrays: dict, lo: int = 0) -> bytes:
    """The bytes an array of (stem, role) pairs emits, from ``lo`` onwards."""
    stems = arrays["stem"][lo:]
    roles = arrays["role"][lo:] if plan.vocab.n_roles > 1 else np.zeros_like(stems)
    return b"".join(
        plan.vocab.spell(int(s), int(r)) for s, r in zip(stems, roles, strict=True)
    )


def _codec(rules, active=ACTIVE, stream=None, folding=False, text=""):
    if folding:
        enc = shrink.fold_encode(text, TOK, HOST)
        ids = enc.ids.astype(np.int64)
        plan = shrink.build(
            HOST, np.union1d(np.unique(ids), active), ids, rules, fold_case=enc.case
        )
    else:
        plan = shrink.build(HOST, active, SKEWED if stream is None else stream, rules)
    return hellaswag.ShrinkCodec(plan, ARM, folding)


# --- the existing path, unmoved -------------------------------------------


def test_the_host_path_is_still_the_encoding_and_nothing_else():
    """The six transplant arms have published numbers (orig-frozen acc_norm
    0.6035, bpe8k-t 0.5389, ...). Their path has to come out of the refactor
    byte-for-byte: the arrays are the arm's own ids and the boundary is the
    context's own length, with no codec anywhere near it."""
    codec = hellaswag.HostCodec(ARM, ARM.heads)
    doc = _doc(" the cat", ["the", " The", "!", " cat"])
    encoded = hellaswag.encode_docs(codec, [doc])

    assert not codec.shrink
    for seq, choice in zip(encoded[0]["seqs"], doc["choices"], strict=True):
        assert seq["arrays"]["stem"].tolist() == (
            ARM.encode(doc["query"] + " " + choice).stem.tolist()
        )
        assert seq["ctx_len"] == len(ARM.encode(doc["query"]).stem)
        assert seq["cont_bytes"] == len((" " + choice).encode("utf-8"))
        # No shrink-only bookkeeping leaks into a published artifact's shape.
        assert set(seq) == {"arrays", "ctx_len", "cont_bytes"}


def test_a_transplant_checkpoint_still_takes_the_host_path():
    """``stage`` is absent from every checkpoint written before the shrink
    wave, so the six published arms must key off its absence."""
    codec = hellaswag.build_codec({"heads": ARM.heads}, ARM, None, None)
    assert isinstance(codec, hellaswag.HostCodec)


# --- the codec, applied to a document -------------------------------------


@pytest.mark.parametrize("k", [1, 2, 3])
def test_a_shrunk_document_round_trips_to_its_own_bytes(k):
    """The statement everything else rests on: the token stream changes, the
    byte string it spells does not. ``cont_bytes`` is ``acc_norm``'s
    denominator, so a stream that spells anything else is normalizing by a
    length it does not have."""
    rules = SHRINK_CURRICULUM[:2] + (ShrinkRule("repiece_tail", min_count=20),)
    codec = _codec(rules[:k])
    doc = _doc(" the cat the", ["the", " The cat", "aBc"])
    encoded = hellaswag.encode_docs(codec, [doc])  # ``check`` runs inside
    for seq, choice in zip(encoded[0]["seqs"], doc["choices"], strict=True):
        assert _spell(codec.plan, seq["arrays"]) == (
            doc["query"] + " " + choice
        ).encode("utf-8")


def test_the_continuation_boundary_is_recomputed_not_carried():
    """The subtle one. Rule 2 turns one source token into several, so a
    boundary counted before the rewrite lands *inside* the context afterwards
    and every choice is scored on a sum that includes context NLL — a number
    that looks entirely reasonable and is not the benchmark.

    The assertion is the strong form: the scored region spells exactly the
    continuation's bytes, not one byte more or less."""
    rules = SHRINK_CURRICULUM[:2] + (ShrinkRule("repiece_tail", min_count=20),)
    codec = _codec(rules)
    doc = _doc(" the cat the", ["the"])
    seq = hellaswag.encode_docs(codec, [doc])[0]["seqs"][0]

    src_ctx_len = len(ARM.encode(doc["query"]).stem)
    assert seq["ctx_len"] > src_ctx_len, "the context should have re-pieced"
    assert _spell(codec.plan, seq["arrays"], seq["ctx_len"]) == b" the"
    # Carrying the boundary would have scored part of the context instead.
    assert _spell(codec.plan, seq["arrays"], src_ctx_len) != b" the"


def test_a_relabeling_rung_leaves_the_boundary_where_it_was():
    """Rules 0 and 1 emit one pair per token, so at those rungs the recomputed
    boundary must land exactly where the carried one would have — otherwise the
    recomputation is introducing an error of its own."""
    codec = _codec(SHRINK_CURRICULUM[:2])
    doc = _doc(" the cat the", ["the", " The"])
    for seq in hellaswag.encode_docs(codec, [doc])[0]["seqs"]:
        assert seq["ctx_len"] == len(ARM.encode(doc["query"]).stem)


def test_a_fold_arm_reads_its_own_encoding_and_round_trips():
    """A ``fold`` arm's stream is not ``arm.encode`` at all: it is the host
    tokenizer over the *lowercased* text with the case read back off the
    original. Scoring it through the merge path would be a different arm."""
    rules = SHRINK_FOLD_CURRICULUM[:2] + (ShrinkRule("repiece_tail", min_count=3),)
    codec = _codec(rules, folding=True, text=" the cat " * 20)
    doc = _doc(" The Cat", [" THE", "the cat", " tHe"])
    for seq, choice in zip(
        hellaswag.encode_docs(codec, [doc])[0]["seqs"], doc["choices"], strict=True
    ):
        assert _spell(codec.plan, seq["arrays"]) == (
            doc["query"] + " " + choice
        ).encode("utf-8")
        assert _spell(codec.plan, seq["arrays"], seq["ctx_len"]) == (
            " " + choice
        ).encode("utf-8")


def test_a_stream_that_does_not_spell_the_text_is_caught():
    """``check`` is what makes the round trip a fact rather than an argument,
    so it has to fire when the arrays stop matching the bytes."""
    codec = _codec(SHRINK_CURRICULUM[:2])
    arrays = codec.document(codec.encode(" the cat"), 1, 4)["arrays"]
    with pytest.raises(ValueError, match="does not round-trip|spells"):
        codec.check(arrays, b" the dog")


# --- what the codec has no row for ----------------------------------------


def test_a_token_with_no_row_is_spelled_out_of_the_rows_that_remain():
    """HellaSwag is English from a different distribution than this ru+en
    corpus: 2,072 of its 3,166,583 host tokens never occur in it, so rule 0
    deleted their rows. A transplant arm scores those unmasked because the row
    still exists; a shrink arm has to re-piece them, which is rule 2 pointed at
    out-of-corpus text."""
    dropped = SURFACES.index(b" cat")
    active = np.array([i for i in ACTIVE.tolist() if i != dropped], dtype=np.int64)
    codec = _codec(
        SHRINK_CURRICULUM[:1], active=active, stream=_stream(active.tolist(), 4)
    )
    doc = _doc(" the", ["cat"] * 4)
    encoded = hellaswag.encode_docs(codec, [doc])

    seq = encoded[0]["seqs"][0]
    assert seq["source_cont_tokens"] == 1
    assert seq["source_cont_repieced"] == 1
    # One source token, more than one scored token, and the same bytes.
    assert int(seq["arrays"]["stem"].size) - seq["ctx_len"] > 1
    assert _spell(codec.plan, seq["arrays"], seq["ctx_len"]) == b" cat"


def test_the_coverage_diagnostic_counts_what_the_codec_cannot_express():
    """For a shrink arm "outside the active vocabulary" is vacuous — the
    inactive rows are deleted, so every id the codec accepts is active by
    construction and the number would read 0.0000 forever while measuring
    nothing. The honest analogue is one rung earlier."""
    dropped = SURFACES.index(b" cat")
    active = np.array([i for i in ACTIVE.tolist() if i != dropped], dtype=np.int64)
    codec = _codec(
        SHRINK_CURRICULUM[:1], active=active, stream=_stream(active.tolist(), 4)
    )
    encoded = hellaswag.encode_docs(codec, [_doc(" the", ["cat"] * 4)])

    cov = hellaswag.codec_coverage(encoded)
    assert cov["mode"] == "codec"
    assert cov["source_continuation_tokens"] == 4
    assert cov["tokens_outside_codec"] == 4
    assert cov["fraction_outside_codec"] == pytest.approx(1.0)
    # Scored tokens exceed source tokens: that gap *is* the re-piecing.
    assert cov["continuation_tokens"] > cov["source_continuation_tokens"]


def test_text_the_arm_cannot_spell_at_all_is_a_stop_not_a_gap():
    """Before a ``repiece_tail`` rung the byte floor is incomplete, so a
    surface can have no segmentation. Dropping it would score fewer bytes than
    ``acc_norm`` divides by, which is exactly the silent kind of wrong."""
    active = np.array(
        [SURFACES.index(b" the"), SURFACES.index(b"the"), SURFACES.index(b" ")],
        dtype=np.int64,
    )
    codec = _codec(
        SHRINK_CURRICULUM[:1], active=active, stream=_stream(active.tolist(), 4)
    )
    with pytest.raises(ValueError, match="cannot be spelled out of"):
        hellaswag.encode_docs(codec, [_doc(" the", ["cat"])])


# --- rebuilding the codec from a checkpoint -------------------------------


def test_an_older_checkpoint_without_a_curriculum_field_reads_as_the_default():
    """``shrink1-case`` and the tail arms were trained before the fold
    curriculum existed, so their checkpoints carry ``stage`` and no
    ``curriculum`` at all. They are the default chain."""
    spec = hellaswag._shrink_spec({"experiment": "shrink1-case", "stage": 2})
    assert (spec.curriculum, spec.stage) == ("default", 2)


def test_the_curricula_are_not_interchangeable_at_the_same_stage():
    """``default`` and ``fold`` share stage numbers and produce different
    vocabularies at each, so a checkpoint that disagrees with its config row is
    a rebuild that would index the wrong table."""
    with pytest.raises(ValueError, match="curriculum"):
        hellaswag._shrink_spec(
            {"experiment": "shrink1-fold", "stage": 2, "curriculum": "default"}
        )


def test_a_stage_checkpoint_with_no_config_row_is_refused():
    """``override_min_count`` is not in the checkpoint — ``shrink2-tail200``
    differs from ``shrink2-tail100`` in nothing else — so configs stays the
    source of truth and a missing row cannot be guessed around."""
    with pytest.raises(KeyError, match="configs.SHRINKS"):
        hellaswag._shrink_spec({"experiment": "shrink9-imaginary", "stage": 3})
