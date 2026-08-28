"""Feature-routed boosted trees as an embedding table.

The shape of the idea: a token's embedding is the sum of one vector per tree in
a boosted ensemble, and which vector it gets is decided by feature tests on
``token_features`` rather than by a row index. Parameters live at shared leaves,
so V tokens cost ``n_leaves x d`` floats instead of ``V x d``.

**The partition is fit offline; the values are not.** Stage 0 fit structure to
the frozen table offline and every variant cost at least +1.37 BPB against a
0.05 kill line — matching the matrix is not matching the function. So the trees
here are fit to mean-of-pieces vectors purely to decide *which tokens share a
leaf*, and every leaf vector is then re-learned through the frozen body. Nothing
in ``fit_partition``'s output is trusted as a value; only ``routing`` survives
into training, and ``leaf_init`` is an init, in the way study 2 measured
mean-of-pieces to be worth 0.61 BPB as an init and nothing more.

**Why the split search runs on projected targets.** sklearn's multi-output
splitter evaluates variance across every output column at every candidate split,
so fitting directly against 2048-dimensional targets costs
``n_features x n_samples x 2048`` per level and takes hours. The targets only
define the partition, so they are projected to ``fit_dim`` first. This is not a
corner cut in the same sense as stage 0's: there, discarded target detail became
frozen *values* that were then wrong; here it can only make the partition
slightly less well chosen, and the vectors are relearned regardless.

**Additive is representationally flat.** Stage 0 established that a depth-2
path-sum equals flat VQ with the same leaf count — an additive ensemble buys
sharing, not expressivity. That is exactly why the FiLM arm exists: composing
``e <- e * scale + shift`` down the ensemble is not a sum, so depth can buy
something a codebook cannot express.

Accounting follows ``distill_embed``: floats are floats, and the discrete route
is per-token *bits*, reported separately and never folded into a compression
ratio. Ratios are quoted against the **active** flat table (see
``active_vocab``), not the full one.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn

from experiments.vocab_factorization.configs import TreeSpec


@dataclass(frozen=True)
class Routing:
    """Which leaf each token reaches in each tree, plus the offline init."""

    table: np.ndarray  # (V, n_trees) int64, already offset into one leaf bank
    n_leaves: int
    leaf_init: np.ndarray  # (n_leaves, d) float32 — an init, never a value
    per_tree_leaves: tuple[int, ...]

    @property
    def vocab(self) -> int:
        return self.table.shape[0]


def _project(targets: np.ndarray, fit_dim: int, seed: int) -> np.ndarray:
    """Reduce targets for the split search only (see the module docstring)."""
    if fit_dim <= 0 or fit_dim >= targets.shape[1]:
        return targets
    centered = targets - targets.mean(0, keepdims=True)
    # A seeded Gaussian projection rather than a PCA: it is O(Vd*k) instead of
    # a full SVD of the table, and Johnson-Lindenstrauss says pairwise
    # geometry — which is all a partition depends on — survives it.
    rng = np.random.default_rng(seed)
    proj = rng.normal(size=(targets.shape[1], fit_dim)).astype(np.float32)
    proj /= np.sqrt(fit_dim)
    return centered @ proj


def fit_partition(features: np.ndarray, targets: np.ndarray, spec: TreeSpec) -> Routing:
    """Boosted residual fit. Returns the routing and a leaf-mean init.

    Each round fits a tree to what the ensemble so far has not explained, which
    is what makes later trees carve the space the earlier ones left coarse.
    """
    from sklearn.tree import DecisionTreeRegressor

    V, d = targets.shape
    fit_target = _project(targets, spec.fit_dim, spec.seed)
    residual = fit_target.copy()

    columns, per_tree, offset = [], [], 0
    assign_full = np.zeros((V, spec.n_trees), dtype=np.int64)
    for t in range(spec.n_trees):
        tree = DecisionTreeRegressor(
            max_depth=spec.max_depth,
            max_leaf_nodes=spec.leaves_per_tree,
            random_state=spec.seed + t,
        )
        tree.fit(features, residual)
        node = tree.apply(features)
        # ``apply`` returns node ids in the tree's internal numbering, which is
        # sparse; remap to a contiguous block so every leaf in the ensemble has
        # exactly one row in one shared bank.
        uniq, local = np.unique(node, return_inverse=True)
        assign_full[:, t] = local + offset
        per_tree.append(len(uniq))
        offset += len(uniq)
        columns.append(local)
        residual -= tree.predict(features).reshape(residual.shape)

    # The init is the leaf mean of the *full-dimensional* targets, computed once
    # the partition is settled: the projection chose the split, it does not get
    # to distort the starting point.
    n_leaves = offset
    leaf_init = np.zeros((n_leaves, d), dtype=np.float32)
    counts = np.zeros(n_leaves, dtype=np.float64)
    share = (targets / spec.n_trees).astype(np.float32)
    np.add.at(
        leaf_init, assign_full.reshape(-1), np.repeat(share, spec.n_trees, axis=0)
    )
    np.add.at(counts, assign_full.reshape(-1), 1.0)
    leaf_init /= np.maximum(counts, 1.0)[:, None].astype(np.float32)

    return Routing(
        table=assign_full,
        n_leaves=n_leaves,
        leaf_init=leaf_init,
        per_tree_leaves=tuple(per_tree),
    )


class TreeEmbedding(nn.Module):
    """Fixed routing, trainable leaves. A drop-in for ``nn.Embedding``.

    ``TransplantLM`` needs two things from a stem embedding: a lookup for the
    input side, and a ``(V, d)`` matrix for the tied output side. ``weight`` is
    a *property* that composes the table from the leaves, so the same module can
    serve both and the tie is structural rather than a shared Parameter —
    exactly the "output embedding composed from the same tied structure" the
    plan requires, and the side stage 0 measured to be the sensitive one.
    """

    def __init__(self, routing: Routing, d_model: int, film: bool = False):
        super().__init__()
        self.vocab, self.n_trees = routing.table.shape
        self.n_leaves = routing.n_leaves
        self.d_model = d_model
        self.film = film
        self.per_tree_leaves = routing.per_tree_leaves
        self.register_buffer(
            "routing", torch.from_numpy(routing.table).long(), persistent=True
        )
        self.leaves = nn.Parameter(torch.from_numpy(routing.leaf_init).float())
        if film:
            # Identity at init, so a FiLM arm starts exactly where the additive
            # one does and any difference is something training found.
            self.film_scale = nn.Parameter(torch.ones(self.n_leaves, d_model))
            self.film_shift = nn.Parameter(torch.zeros(self.n_leaves, d_model))

    def _compose(self, routes: torch.Tensor) -> torch.Tensor:
        """Compose embeddings for rows whose per-tree leaf ids are ``routes``.

        Accumulated one tree at a time, deliberately. The obvious
        ``leaves[routes]`` gathers ``(n, n_trees, d)`` in one shot, which for
        the host bridge arm is 151,936 x 48 x 2048 floats — 59 GB, before any
        activation. Gathering per tree keeps the peak at a couple of ``(n, d)``
        tensors, and addition's backward needs neither operand, so the partial
        sums are freed as the loop advances rather than retained by autograd.

        The FiLM path cannot enjoy that: a product's backward *does* need both
        operands, so its intermediates are retained and it costs ``n_trees``
        times as much. That is affordable only because FiLM is an 8k-vocabulary
        arm; it is the reason there is no FiLM arm on the 152k host.
        """
        out = self.leaves[routes[..., 0]]
        for t in range(1, self.n_trees):
            if self.film:
                out = (
                    out * self.film_scale[routes[..., t]]
                    + self.film_shift[routes[..., t]]
                )
            else:
                out = out + self.leaves[routes[..., t]]
        return out

    @property
    def weight(self) -> torch.Tensor:
        """The composed ``(V, d)`` table, for the tied output side."""
        return self._compose(self.routing)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # Composed for the batch's ids only. Going through ``weight`` would
        # build the whole table for a few thousand lookups.
        return self._compose(self.routing[ids])

    def float_params(self) -> int:
        n = self.n_leaves * self.d_model
        return n * 3 if self.film else n

    def index_bits(self) -> int:
        """Route storage, in bits per token summed over trees. Never a float."""
        return self.vocab * sum(
            math.ceil(math.log2(max(2, k))) for k in self.per_tree_leaves
        )

    def accounting(self, flat_floats: int) -> dict:
        floats = self.float_params()
        return {
            "n_leaves": self.n_leaves,
            "n_trees": self.n_trees,
            "film": self.film,
            "float_params": floats,
            "index_bits": self.index_bits(),
            "flat_floats": flat_floats,
            "compression": flat_floats / floats,
        }


def build(
    features: np.ndarray,
    targets: np.ndarray,
    spec: TreeSpec,
    d_model: int | None = None,
) -> TreeEmbedding:
    routing = fit_partition(features, targets, spec)
    return TreeEmbedding(routing, d_model or targets.shape[1], film=spec.film)


def leaves_for_budget(
    flat_floats: int, ratio: float, d_model: int, n_trees: int, film: bool = False
) -> int:
    """Leaves per tree that land an arm at ``ratio`` fewer floats than flat.

    Written as a function because the budget is the *claim* — "parity at >=x2
    fewer embedding floats" — so it should be derived from the flat table the
    arm is actually measured against rather than typed in and drifting from it.
    """
    # A FiLM leaf stores three vectors (value, scale, shift), so hitting the
    # same float budget means a third of the leaves. Charging it honestly is
    # what keeps "at x2 fewer floats" the same claim for both arms.
    per_leaf = d_model * (3 if film else 1)
    total = flat_floats / (ratio * per_leaf)
    return max(2, int(total // n_trees))
