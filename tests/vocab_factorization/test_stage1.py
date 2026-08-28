"""Stage-1 routing: features, tree structure, and the accounting the claim rests on.

The claim stage 1 makes is arithmetic — "parity BPB at >=x2 fewer embedding
floats" — so the float count is not bookkeeping, it *is* the result. A leaf bank
counted once when it is used twice, or a FiLM arm charged for one vector when it
stores three, would move the compression ratio without moving anything real.
Those are the assertions here, along with the two structural properties the
design depends on: routing that cannot drift during training, and leaves that
can.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

from experiments.vocab_factorization import (  # noqa: E402
    token_features,
    transplant,
    tree_embed,
)
from experiments.vocab_factorization.configs import FeatureSpec, TreeSpec  # noqa: E402
from tests.vocab_factorization.test_w0 import D_MODEL, VOCAB, StubBody  # noqa: E402


class FakeArm:
    """Just enough arm for the feature extractor: surfaces and a head count."""

    kind = "bpe"

    def __init__(self, surfaces):
        self._s = list(surfaces)
        self.heads = {"stem": len(self._s)}

        class _Tok:
            def decode(_, ids, **kw):
                return self._s[ids[0]]

        self.tok = _Tok()


SURFACES = [
    " hello",
    "WORLD",
    "Привет",
    "42",
    "!!",
    " Thing",
    "мир",
    "aBc",
]


def _features(groups=("surface", "ngram")):
    return token_features.extract(
        FakeArm(SURFACES), FeatureSpec(groups=groups, ngram_buckets=32)
    )


# --- features -------------------------------------------------------------


def test_extraction_is_deterministic():
    """The evolution phase mutates this module; a feature that depended on call
    order would turn its fitness signal into noise."""
    a, b = _features(), _features()
    assert np.array_equal(a.matrix, b.matrix)
    assert a.names == b.names


def test_surface_features_say_what_they_mean():
    f = _features(("surface",))
    col = {n: i for i, n in enumerate(f.names)}
    m = f.matrix
    assert m[0, col["flag.leading_space"]] == 1.0  # " hello"
    assert m[1, col["flag.leading_space"]] == 0.0  # "WORLD"
    assert m[1, col["case.upper"]] == 1.0
    assert m[0, col["case.lower"]] == 1.0
    assert m[5, col["case.title"]] == 1.0  # " Thing"
    assert m[7, col["case.mixed"]] == 1.0  # "aBc"
    assert m[2, col["script.cyrillic"]] == 1.0  # "Привет"
    assert m[0, col["script.latin"]] == 1.0
    assert m[3, col["script.digit"]] == 1.0  # "42"


def test_groups_are_introspectable_and_ablatable():
    """ladder-calibration needs group ablations and deliberately-bad anchors."""
    f = _features(("surface", "ngram"))
    assert set(f.group_names) == {"surface", "ngram"}
    assert (
        f.select("surface").matrix.shape[1] + f.select("ngram").matrix.shape[1]
        == (f.matrix.shape[1])
    )
    assert f.without("ngram").group_names == ("surface",)
    # The length-only anchor from the plan: one column, still a valid router.
    assert f.columns("len.bytes.log").matrix.shape == (len(SURFACES), 1)
    with pytest.raises(KeyError):
        f.select("corpus")


def test_corpus_stats_are_finite_and_sized_to_the_vocabulary():
    rng = np.random.default_rng(0)
    stream = rng.integers(0, 8, size=5000)
    stats = token_features.corpus_stats(stream, 8)
    for name, arr in stats.items():
        assert arr.shape == (8,), name
        assert np.isfinite(arr).all(), name


# --- tree structure -------------------------------------------------------


def _routing(v=64, f=6, d=16, n_trees=4, leaves=8, seed=0):
    rng = np.random.default_rng(seed)
    feats = rng.normal(size=(v, f)).astype(np.float32)
    targets = rng.normal(size=(v, d)).astype(np.float32)
    spec = TreeSpec(n_trees=n_trees, leaves_per_tree=leaves, fit_dim=4)
    return tree_embed.fit_partition(feats, targets, spec), targets


def test_routing_is_a_buffer_and_leaves_are_parameters():
    """Fixed routing is the design: the partition is chosen offline and the
    values are learned. A routing that drifted under the optimizer would make
    the arm something other than what is being claimed."""
    routing, _ = _routing()
    m = tree_embed.TreeEmbedding(routing, 16)
    names = {n for n, _ in m.named_parameters()}
    assert names == {"leaves"}
    assert "routing" in dict(m.named_buffers())
    assert not m.routing.dtype.is_floating_point


def test_gradient_reaches_the_leaves_through_the_composed_table():
    routing, _ = _routing()
    m = tree_embed.TreeEmbedding(routing, 16)
    m.weight.sum().backward()
    assert m.leaves.grad is not None
    assert torch.count_nonzero(m.leaves.grad) > 0


def test_lookup_and_table_agree():
    """forward() composes only the batch's rows; weight composes all of them.
    They are different code paths and must not drift apart."""
    routing, _ = _routing()
    m = tree_embed.TreeEmbedding(routing, 16)
    ids = torch.tensor([[0, 5, 5], [63, 1, 2]])
    assert torch.allclose(m(ids), m.weight[ids], atol=1e-6)


def test_composition_never_materializes_the_full_gather():
    """A (V, n_trees, d) gather is 59 GB for the host bridge arm, so the
    composition has to accumulate per tree instead."""
    routing, _ = _routing(v=64, d=16, n_trees=4, leaves=8)
    m = tree_embed.TreeEmbedding(routing, 16)
    peak = {"n": 0}
    real_getitem = torch.Tensor.__getitem__

    def spy(self, idx):
        out = real_getitem(self, idx)
        if out.dim() >= 3:
            peak["n"] = max(peak["n"], out.numel())
        return out

    torch.Tensor.__getitem__ = spy
    try:
        m.weight.sum().backward()
    finally:
        torch.Tensor.__getitem__ = real_getitem
    assert peak["n"] == 0, "a 3-D gather means the whole (V, n_trees, d) block"


def test_every_token_routes_within_the_leaf_bank():
    routing, _ = _routing()
    assert routing.table.min() >= 0
    assert routing.table.max() < routing.n_leaves
    assert sum(routing.per_tree_leaves) == routing.n_leaves


def test_film_stores_three_vectors_per_leaf_and_is_charged_for_them():
    """The FiLM arm must not buy expressivity by quietly spending more floats."""
    routing, _ = _routing()
    add = tree_embed.TreeEmbedding(routing, 16, film=False)
    film = tree_embed.TreeEmbedding(routing, 16, film=True)
    assert film.float_params() == 3 * add.float_params()

    flat = 64 * 16
    n_add = tree_embed.leaves_for_budget(flat, 2.0, 16, 4, film=False)
    n_film = tree_embed.leaves_for_budget(flat, 2.0, 16, 4, film=True)
    assert n_film * 3 == pytest.approx(n_add, rel=0.35)


def test_film_starts_as_the_additive_arm():
    """scale=1, shift=0 at init, so any difference is something training found."""
    routing, _ = _routing()
    add = tree_embed.TreeEmbedding(routing, 16, film=False)
    film = tree_embed.TreeEmbedding(routing, 16, film=True)
    with torch.no_grad():
        film.leaves.copy_(add.leaves)
    # Additive sums all trees; FiLM at identity keeps only the first tree's leaf,
    # so they are *not* equal — the assertion is that FiLM is the identity
    # transform on its base, not that it duplicates the additive arm.
    expected = film.leaves[film.routing[:, 0]]
    assert torch.allclose(film.weight, expected, atol=1e-6)


def test_accounting_hits_the_budget_and_keeps_bits_out_of_it():
    routing, _ = _routing(v=64, leaves=8, n_trees=4)
    m = tree_embed.TreeEmbedding(routing, 16)
    flat = 64 * 16
    acct = m.accounting(flat)
    assert acct["float_params"] == routing.n_leaves * 16
    assert acct["compression"] == pytest.approx(flat / acct["float_params"])
    # Route storage is bits, reported beside the floats and never inside them.
    assert acct["index_bits"] > 0
    assert acct["index_bits"] not in (acct["float_params"], acct["compression"])


# --- the swap into TransplantLM -------------------------------------------


def test_the_tie_is_structural_and_counted_once():
    """Both sides are the same module, so the output table is composed from the
    same leaves. param_counts has to notice that by module identity: the
    composed ``weight`` is a fresh tensor on every access, so the old
    ``is``-on-tensor test would silently double every leaf."""
    routing, _ = _routing(v=VOCAB, d=D_MODEL, leaves=8, n_trees=4)
    model = transplant.TransplantLM(StubBody(), {"stem": VOCAB}, D_MODEL)
    flat_before = model.param_counts()["embedding"]

    tree = tree_embed.TreeEmbedding(routing, D_MODEL)
    model.emb["stem"] = tree
    model.out["stem"] = tree

    counts = model.param_counts()
    assert model.out["stem"] is model.emb["stem"]
    assert counts["embedding"] == tree.float_params()
    assert counts["output_heads"] == 0, "the tied routed table was counted twice"
    assert counts["embedding"] < flat_before


def test_a_routed_model_still_trains_and_scores():
    """End to end on CPU: the swapped module has to satisfy both contracts —
    lookup on the input side, a (V, d) matrix on the output side."""
    routing, _ = _routing(v=VOCAB, d=D_MODEL, leaves=8, n_trees=4)
    model = transplant.TransplantLM(StubBody(), {"stem": VOCAB}, D_MODEL)
    tree = tree_embed.TreeEmbedding(routing, D_MODEL)
    model.emb["stem"] = tree
    model.out["stem"] = tree

    ids = torch.randint(0, VOCAB, (2, 5))
    loss = model({"stem": ids[:, :-1]}, {"stem": ids[:, 1:]})["stem"]
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.count_nonzero(tree.leaves.grad) > 0
