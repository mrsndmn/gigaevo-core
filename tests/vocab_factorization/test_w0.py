"""W0 mechanisms, exercised without a GPU or a 1.7B download.

Both things W0 added are cheap to get subtly wrong and expensive to discover
wrong: a softmax mask that does not actually restrict the normalizer would
silently produce numbers that look fine and are not comparable, and a
checkpoint whose keys do not survive a reload would only be discovered the
next time somebody needed the weights — which is the exact failure study 2
already paid for once.

A stub body stands in for Qwen3: ``TransplantLM`` only ever asks it for
``last_hidden_state``, so nothing here depends on the real model.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from experiments.vocab_factorization import active_vocab, transplant  # noqa: E402

D_MODEL = 8
VOCAB = 64
SEQ = 6
BATCH = 2


class StubBody(torch.nn.Module):
    """Identity-ish body: one trainable parameter, returns hidden states."""

    def __init__(self, d_model: int = D_MODEL):
        super().__init__()
        self.proj = torch.nn.Linear(d_model, d_model, bias=False)
        self.embed_tokens = torch.nn.Embedding(VOCAB, d_model)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, inputs_embeds=None, **_):
        return types.SimpleNamespace(last_hidden_state=self.proj(inputs_embeds))


def _model(heads=None) -> transplant.TransplantLM:
    torch.manual_seed(0)
    return transplant.TransplantLM(StubBody(), heads or {"stem": VOCAB}, D_MODEL)


def _batch(active_ids: np.ndarray, seed: int = 0):
    """Inputs and targets drawn only from ids the mask keeps."""
    g = torch.Generator().manual_seed(seed)
    pick = torch.from_numpy(active_ids)[
        torch.randint(0, len(active_ids), (BATCH, SEQ + 1), generator=g)
    ]
    return {"stem": pick[:, :-1]}, {"stem": pick[:, 1:]}


# --- the mask -------------------------------------------------------------


def test_mask_removes_competitors_from_the_normalizer():
    """Masking can only lower NLL: it deletes terms from the denominator."""
    active = np.arange(0, VOCAB, 2)  # every other id
    model = _model()
    inputs, targets = _batch(active)

    unmasked = float(model(inputs, targets)["stem"].detach())
    model.set_active_mask(active_vocab.mask_from_ids(active, VOCAB))
    masked = float(model(inputs, targets)["stem"].detach())

    assert masked < unmasked
    # Half the vocabulary removed is a real change, not a rounding artifact.
    assert unmasked - masked > 1e-3


def test_mask_zeroes_the_gradient_of_inactive_output_rows():
    """Convention (b) — "inactive tokens are not optimized" — for free."""
    active = np.arange(0, VOCAB, 2)
    model = _model()
    model.set_active_mask(active_vocab.mask_from_ids(active, VOCAB))
    inputs, targets = _batch(active)

    model(inputs, targets)["stem"].backward()
    grad = model.emb["stem"].weight.grad
    inactive = np.setdiff1d(np.arange(VOCAB), active)
    assert torch.count_nonzero(grad[inactive]) == 0
    assert torch.count_nonzero(grad[active]) > 0


def test_mask_applies_on_the_chunked_path_too():
    """The large-vocabulary branch recomputes logits in backward under
    ``checkpoint``; the mask has to survive that, and it is the branch every
    real host-vocabulary run takes."""
    active = np.arange(0, VOCAB, 2)
    inputs, targets = _batch(active)
    mask = active_vocab.mask_from_ids(active, VOCAB)

    plain = _model()
    plain.set_active_mask(mask)
    expected = float(plain(inputs, targets)["stem"].detach())

    chunked = _model()
    chunked.set_active_mask(mask)
    old_threshold, old_chunk = transplant._CHUNK_VOCAB_THRESHOLD, transplant._CE_CHUNK
    transplant._CHUNK_VOCAB_THRESHOLD, transplant._CE_CHUNK = 1, 3
    try:
        got = float(chunked(inputs, targets)["stem"].detach())
    finally:
        transplant._CHUNK_VOCAB_THRESHOLD = old_threshold
        transplant._CE_CHUNK = old_chunk

    assert got == pytest.approx(expected, rel=1e-5)


def test_mask_must_match_the_table_not_the_tokenizer():
    """An HF host's table is padded past ``len(tokenizer)``; a mask sized to the
    tokenizer would silently misalign every row after the first gap."""
    model = _model()
    with pytest.raises(ValueError, match="padded row count"):
        model.set_active_mask(torch.ones(VOCAB - 3, dtype=torch.bool))


def test_inactive_targets_are_refused_rather_than_scored_at_zero():
    active = np.arange(0, VOCAB, 2)
    arrays = {"stem": np.array([0, 2, 4, 7], dtype=np.int64)}  # 7 is inactive
    with pytest.raises(ValueError, match="outside the active vocabulary"):
        active_vocab.assert_targets_active(arrays, active, "unit/clean")


# --- the checkpoint -------------------------------------------------------


def test_checkpoint_round_trips_the_parameters(tmp_path):
    """What ``train_transplant`` writes is what ``eval_transplant`` reads."""
    from experiments.vocab_factorization.configs import (
        TRANSPLANTS_BY_NAME,
        TransplantTrainSpec,
    )
    from experiments.vocab_factorization.train_transplant import _save_checkpoint

    heads = {"stem": VOCAB, "role": 4}
    model = _model(heads)
    with torch.no_grad():  # make the saved weights distinctive
        model.emb["stem"].weight.normal_()
        model.emb["role"].weight.normal_()

    path = tmp_path / "arm.pt"
    _save_checkpoint(
        model,
        TRANSPLANTS_BY_NAME["bpe8k-t"],
        TransplantTrainSpec(),
        heads,
        steps=7,
        path=path,
    )

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert ckpt["heads"] == heads
    assert ckpt["steps"] == 7

    restored = _model(heads)
    restored.emb.load_state_dict(ckpt["emb"])
    restored.out.load_state_dict(ckpt["out"])
    for name in heads:
        assert torch.equal(restored.emb[name].weight, model.emb[name].weight), (
            f"{name} embedding did not survive the round trip"
        )
        assert torch.equal(restored.out[name].weight, model.out[name].weight)

    # The tie has to survive too: an untied reload would double the parameters
    # and silently break the "output side is composed from the same table" claim.
    assert restored.out["stem"].weight is restored.emb["stem"].weight


# --- the unfrozen-body arm (W1) -------------------------------------------


def test_freeze_body_controls_what_trains():
    frozen = transplant.TransplantLM(StubBody(), {"stem": VOCAB}, D_MODEL)
    thawed = transplant.TransplantLM(
        StubBody(), {"stem": VOCAB}, D_MODEL, freeze_body=False
    )
    assert not any(p.requires_grad for p in frozen.body.parameters())
    assert all(p.requires_grad for p in thawed.body.parameters())
    assert len(thawed.trainable_parameters()) > len(frozen.trainable_parameters())


def test_only_an_unfrozen_arm_stores_the_body(tmp_path):
    """A frozen body is bit-identical to HF, so storing it wastes 3.4 GB; an
    unfrozen one is the thing that changed, so omitting it would reload a
    different model than the one that was scored."""
    from dataclasses import replace

    from experiments.vocab_factorization.configs import (
        TRANSPLANTS_BY_NAME,
        TransplantTrainSpec,
    )
    from experiments.vocab_factorization.train_transplant import _save_checkpoint

    heads = {"stem": VOCAB}
    tspec = TransplantTrainSpec()
    spec = TRANSPLANTS_BY_NAME["bpe8k-t"]
    assert spec.freeze_body

    frozen_path = tmp_path / "frozen.pt"
    _save_checkpoint(_model(heads), spec, tspec, heads, 1, frozen_path)
    assert "body" not in torch.load(frozen_path, weights_only=False)

    thawed = transplant.TransplantLM(StubBody(), heads, D_MODEL, freeze_body=False)
    thawed_path = tmp_path / "thawed.pt"
    _save_checkpoint(
        thawed, replace(spec, freeze_body=False), tspec, heads, 1, thawed_path
    )
    blob = torch.load(thawed_path, weights_only=False)
    assert "body" in blob and blob["freeze_body"] is False


def test_an_unfrozen_checkpoint_without_a_body_is_refused(tmp_path):
    """The dangerous case: it would otherwise load the pristine HF body next to
    an embedding trained against a different one, and report that as the score."""
    from experiments.vocab_factorization.configs import HOST_MODEL
    from experiments.vocab_factorization.eval_transplant import load_checkpoint

    path = tmp_path / "bad.pt"
    torch.save(
        {"host": HOST_MODEL, "freeze_body": False, "heads": {"stem": VOCAB}}, path
    )
    with pytest.raises(ValueError, match="no body weights"):
        load_checkpoint(path, "cpu")
