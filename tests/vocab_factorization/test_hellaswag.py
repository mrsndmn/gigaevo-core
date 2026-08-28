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

from experiments.vocab_factorization import hellaswag, transplant  # noqa: E402
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
