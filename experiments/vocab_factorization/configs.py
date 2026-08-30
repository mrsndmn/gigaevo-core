"""Experiment grid, as code. This module is the single source of truth.

Two levels:

``ARMS``        tokenizer arms -- what gets built and screened without training
``EXPERIMENTS`` training runs  -- one small LM per row of the ablation grid

Nothing about the grid lives in the plan markdown; the markdown points here.

This module deliberately imports **nothing but the standard library**. The
MLSpace launchers (``jobs.py``, ``run_train_jobs.py``) run under the ``arkhip``
environment, which is where ``mls`` is installed and which has neither
``regex`` nor ``transformers``; they read the grid and the on-disk cache
conventions from here, so keeping this file dependency-free is what lets one
process decide *what to launch* and another actually run it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Tokenizer arms
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmSpec:
    name: str
    kind: str  # bpe | byte | factorized
    vocab_size: int = 0
    n_suffix: int = 256
    use_suffix: bool = True
    # Per-run case classification for mixed-case words instead of dropping the
    # whole word to the verbatim byte path. False reproduces the original arms.
    piece_case: bool = False
    note: str = ""


ARMS: list[ArmSpec] = [
    ArmSpec("bpe32k", "bpe", 32000, note="reference tokenizer"),
    ArmSpec("bpe16k", "bpe", 16000, note="null hypothesis: just shrink the vocab"),
    ArmSpec("bpe8k", "bpe", 8000, note="null hypothesis, smaller still"),
    ArmSpec("byte", "byte", 256, note="robustness ceiling / compression floor"),
    ArmSpec("fact16k", "factorized", 16000, note="space + case + suffix heads"),
    ArmSpec("fact8k", "factorized", 8000, note="space + case + suffix heads"),
    ArmSpec(
        "fact16k-nosuf",
        "factorized",
        16000,
        use_suffix=False,
        note="ablation: space + case only, isolates the morphology head",
    ),
    ArmSpec(
        "fact8k-nosuf",
        "factorized",
        8000,
        use_suffix=False,
        note="space + case only at half the vocab -- added after the first "
        "grid showed vocab size barely moves BPB while the suffix head costs "
        "a lot, which makes this the configuration the results point at",
    ),
    # --- round 2: the two fixes, isolated then combined ---------------------
    # Both are rebuilt with the corrected stem-BPE training (natural span
    # frequency instead of min(f,32)-capped types). fact8k-nosuf above keeps
    # the old cached tokenizer, so it remains the honest before-picture.
    ArmSpec(
        "fact8k-freqfix",
        "factorized",
        8000,
        use_suffix=False,
        piece_case=False,
        note="fix 1 only: stem BPE trained at natural span frequency",
    ),
    ArmSpec(
        "fact8k-v2",
        "factorized",
        8000,
        use_suffix=False,
        piece_case=True,
        note="fix 1 + fix 2: natural-frequency BPE and per-run case, which "
        "removes the verbatim fallback that made randcase the worst suite",
    ),
]

ARMS_BY_NAME = {a.name: a for a in ARMS}

# Bytes of training text used to fit the tokenizers. 50 MB is far more than a
# 32k vocabulary needs and keeps arm construction to a couple of minutes.
TOKENIZER_TRAIN_BYTES = 50 * 1024 * 1024


# ---------------------------------------------------------------------------
# On-disk cache conventions
# ---------------------------------------------------------------------------
#
# Where the shared, read-through caches live. Both the code that *builds* them
# (``prepare``, ``active_vocab``) and the launcher that must check they exist
# before fanning out jobs that would otherwise race to write them agree on
# these two functions rather than on a format string typed twice.


def token_cache_path(
    tokens_dir: Path, arm_name: str, split: str, nbytes: int | None
) -> Path:
    tag = "full" if nbytes is None else f"{nbytes // (1024 * 1024)}mb"
    return tokens_dir / arm_name / f"{split}.{tag}.npz"


def active_cache_path(tokens_dir: Path, arm_name: str) -> Path:
    return tokens_dir / arm_name / "active_vocab.npz"


# The case-folded encoding is a *different* stream over the same arm — host ids
# of the lowercased text plus the case each token restores — so it needs its own
# cache keys rather than a variant spelling of the ones above. Same reason the
# functions exist at all: the launcher checks the cache before fanning out jobs
# that would otherwise race to write it.


def fold_cache_path(
    tokens_dir: Path, arm_name: str, split: str, nbytes: int | None
) -> Path:
    tag = "full" if nbytes is None else f"{nbytes // (1024 * 1024)}mb"
    return tokens_dir / arm_name / f"{split}.{tag}.fold.npz"


def fold_active_cache_path(tokens_dir: Path, arm_name: str) -> Path:
    return tokens_dir / arm_name / "active_vocab.fold.npz"


# The masked native reference (W0, ``results/transplant/orig-frozen.json``):
# clean BPB of the untouched pretrained model under the active-vocab mask. Every
# progressive-shrink rung is gated against it, so it lives here rather than in a
# comment somewhere — and the noise floor is 0.0016 BPB, so the +0.01 gate is
# ~6x noise, not a rounding argument.
MASKED_NATIVE_BPB = 0.7796
SHRINK_GATE_BPB = 0.01


# ---------------------------------------------------------------------------
# Model / training grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    d_model: int = 512
    n_layer: int = 8
    n_head: int = 8
    d_ff: int = 2048
    dropout: float = 0.0
    tie_embeddings: bool = True
    # Character n-gram bag added to the input embedding. This is the mechanism
    # aimed at typos/OOV; it is orthogonal to the case/suffix factorization and
    # is therefore ablated independently.
    ngram_bag: bool = False
    ngram_buckets: int = 8192
    ngram_rank: int = 64
    ngram_min: int = 3
    ngram_max: int = 5
    # Condition the feature heads on the stem being predicted, giving a proper
    # P(stem|h) * P(role|h,stem) * P(suffix|h,stem) factorization rather than
    # assuming the heads are independent given h.
    condition_on_stem: bool = True


@dataclass(frozen=True)
class TrainSpec:
    # DATA is held fixed across arms -- every arm trains on the same
    # ``train_bytes`` of text for the same number of epochs. Budgeting tokens
    # instead would hand a finer tokenizer more optimizer steps over less text,
    # which confounds the representation with the training schedule.
    #
    # The cost of this choice: COMPUTE is *not* equal across arms. The byte arm
    # emits ~5x the tokens of bpe32k for the same text and therefore takes ~5x
    # the compute. Every run records its own steps/tokens/wall-clock so the
    # asymmetry is visible rather than buried.
    train_bytes: int = 400 * 1024 * 1024
    epochs: float = 3.0
    seq_len: int = 1024
    batch_tokens: int = 131072  # tokens per optimizer step
    lr: float = 3e-4
    min_lr_frac: float = 0.1
    warmup_frac: float = 0.02
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    seed: int = 0
    eval_bytes: int = 2 * 1024 * 1024
    log_every: int = 50


@dataclass(frozen=True)
class Experiment:
    name: str
    arm: str
    model: ModelSpec = field(default_factory=ModelSpec)
    train: TrainSpec = field(default_factory=TrainSpec)
    note: str = ""


def _exp(name: str, arm: str, note: str = "", **model_kw) -> Experiment:
    return Experiment(name=name, arm=arm, model=ModelSpec(**model_kw), note=note)


# Order matters only for ``--all``: the head-to-head pair that carries the
# main claim runs first, and the byte arm runs last because it costs ~5x the
# compute of everything else.
EXPERIMENTS: list[Experiment] = [
    # --- the main comparison, at matched vocabulary ------------------------
    _exp("bpe16k", "bpe16k", "null hypothesis at matched vocab"),
    _exp("fact16k", "fact16k", "space + case + suffix"),
    # --- reference and the rest of the vocab sweep -------------------------
    _exp("bpe32k", "bpe32k", "reference"),
    _exp("fact16k-nosuf", "fact16k-nosuf", "ablation: no morphology head"),
    _exp("fact8k", "fact8k", "space + case + suffix, half the stems"),
    _exp("bpe8k", "bpe8k", "null hypothesis, smaller"),
    # --- character n-gram bag (the typo mechanism) -------------------------
    _exp("fact16k-ngram", "fact16k", "full proposal", ngram_bag=True),
    _exp("bpe16k-ngram", "bpe16k", "n-gram bag on a plain BPE", ngram_bag=True),
    # --- robustness ceiling / compression floor ----------------------------
    _exp("byte", "byte", "robustness ceiling"),
    # --- follow-up: the config the first grid's results point at -----------
    _exp("fact8k-nosuf", "fact8k-nosuf", "half vocab, case+space, no suffix head"),
    # --- round 2: the two tokenizer fixes ----------------------------------
    _exp("fact8k-freqfix", "fact8k-freqfix", "fix 1: natural-frequency stem BPE"),
    _exp("fact8k-v2", "fact8k-v2", "fix 1 + fix 2: + per-run case classification"),
]

EXPERIMENTS_BY_NAME = {e.name: e for e in EXPERIMENTS}


# ---------------------------------------------------------------------------
# Study 2: vocabulary transplant onto a frozen pretrained body
# ---------------------------------------------------------------------------

HOST_MODEL = "Qwen/Qwen3-1.7B"
HOST_ARM = "qwen"  # the host's native tokenizer, wrapped as an HFArm


@dataclass(frozen=True)
class TransplantTrainSpec:
    # Same fixed-bytes protocol as study 1. 230 MB is chosen so the *host's
    # native* tokenizer sees ~50M tokens (measured 0.2169 tok/byte on this
    # corpus); the other arms land at 55-57M on the same text.
    train_bytes: int = 230 * 1024 * 1024
    seq_len: int = 1024
    batch_tokens: int = 65536
    # Higher than study 1: only the embedding/head parameters train, and the
    # random-init arms have to move a long way in ~760 steps.
    lr: float = 1e-3
    min_lr_frac: float = 0.1
    warmup_frac: float = 0.05
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    seed: int = 0
    log_every: int = 25


@dataclass(frozen=True)
class TransplantSpec:
    name: str
    arm: str
    init: str  # native | random | mean_of_pieces
    train: bool = True
    micro_batch: int = 8
    # Freezing every transformer weight is what makes the study's claim a
    # *retrofit* claim. One arm lifts it; see ``bpe8k-t-unfrozen``.
    freeze_body: bool = True
    note: str = ""


# ``orig-reinit`` is the load-bearing control: without it, a gap between a
# transplant and the pretrained model conflates "different vocabulary" with
# "embeddings trained for an hour instead of trillions of tokens".
#
# The init pairing is deliberate. orig-reinit and bpe8k-t-random are both
# random, so they compare cleanly; bpe8k-t and fact8k-nosuf-t are both
# mean-of-pieces, so *they* compare cleanly; and bpe8k-t vs bpe8k-t-random
# isolates the init itself.
TRANSPLANTS: list[TransplantSpec] = [
    TransplantSpec(
        "orig-frozen",
        HOST_ARM,
        "native",
        train=False,
        micro_batch=8,
        note="pretrained model untouched — gold reference, eval only",
    ),
    TransplantSpec(
        "orig-reinit",
        HOST_ARM,
        "random",
        micro_batch=8,
        note="native vocab, embeddings retrained from scratch — THE control",
    ),
    TransplantSpec(
        "bpe8k-t-random",
        "bpe8k",
        "random",
        micro_batch=16,
        note="matched-init comparison against orig-reinit",
    ),
    TransplantSpec(
        "bpe8k-t",
        "bpe8k",
        "mean_of_pieces",
        micro_batch=16,
        note="null hypothesis: just transplant a smaller BPE",
    ),
    TransplantSpec(
        "fact8k-nosuf-t",
        "fact8k-nosuf",
        "mean_of_pieces",
        micro_batch=16,
        note="the proposal: 8k stems + case/space role head",
    ),
    # --- transplant-hardening (W1) ------------------------------------------
    # The frozen arms answer "can a vocabulary be retrofitted?". This one asks
    # what the residual gap is made of: with the body free to move, does the
    # transplant close it, and does downstream quality follow BPB or diverge
    # from it? It is the only arm here that trains the body, which is why it
    # gets a quarter of bpe8k-t's micro-batch: AdamW over 1.7B parameters costs
    # ~13.6 GB of moments and ~6.8 GB of fp32 gradients on top of everything
    # the frozen arms already hold, and the activation cost is unchanged
    # because gradients already traversed the whole body to reach layer 0.
    TransplantSpec(
        "bpe8k-t-unfrozen",
        "bpe8k",
        "mean_of_pieces",
        micro_batch=4,
        freeze_body=False,
        note="hardening: bpe8k-t with the body unfrozen, so the clean-BPB gap "
        "can close — isolates 'the embedding is wrong' from 'the body will "
        "not move'",
    ),
]

TRANSPLANTS_BY_NAME = {t.name: t for t in TRANSPLANTS}


# ---------------------------------------------------------------------------
# Study 4, stage 1: feature-routed trees trained through the frozen body
# ---------------------------------------------------------------------------
#
# The specs live here rather than beside their implementations because this
# module is the one the MLSpace launchers can import: they run under an env with
# ``mls`` but without numpy, torch or sklearn. ``token_features`` and
# ``tree_embed`` read their configuration from here.


FEATURE_GROUPS_V0: tuple[str, ...] = ("surface", "decomposition", "ngram", "corpus")


@dataclass(frozen=True)
class FeatureSpec:
    """Which interpretable token features route the tree (see token_features)."""

    groups: tuple[str, ...] = FEATURE_GROUPS_V0
    ngram_buckets: int = 256
    ngram_min: int = 3
    ngram_max: int = 5
    # One-hot rather than a raw bucket index: a tree splits on ``x <= t``, so an
    # arbitrary integer piece id would only let it separate *ranges* of
    # unrelated pieces.
    piece_buckets: int = 32


@dataclass(frozen=True)
class TreeSpec:
    """Structure of one routed table (see tree_embed).

    ``leaves_per_tree = 0`` means "derive it from the arm's budget ratio", which
    keeps the float budget tied to the flat table each arm is actually measured
    against instead of a number typed here that drifts from it.
    """

    n_trees: int = 32
    leaves_per_tree: int = 0
    max_depth: int = 12
    # Targets are projected to this width for the split search only; sklearn's
    # multi-output splitter is O(features x samples x outputs) per level, so
    # 2048-wide targets would take hours to fit and the targets only choose the
    # partition anyway.
    fit_dim: int = 32
    film: bool = False
    seed: int = 0


@dataclass(frozen=True)
class TreeTransplantSpec:
    name: str
    arm: str
    # x fewer embedding floats than the arm's *active* flat table. This is the
    # claim's units: "parity BPB at >=x2 fewer embedding floats".
    budget_ratio: float
    tree: TreeSpec = field(default_factory=TreeSpec)
    features: FeatureSpec = field(default_factory=FeatureSpec)
    micro_batch: int = 16
    # Execution fields with the same names and meanings as ``TransplantSpec``,
    # so ``train_transplant.run`` drives either kind without branching on type.
    init: str = "mean_of_pieces"
    train: bool = True
    freeze_body: bool = True
    # Fit each boosting round to the loss gradient w.r.t. token embeddings
    # collected through the frozen body, instead of to matrix residuals. This is
    # the arm that boosts on the true objective — precisely the correction stage
    # 0's failure demands — and it is optional because it costs extra passes.
    functional_boost: bool = False
    note: str = ""


# The flat control this grid is measured against is ``TRANSPLANTS['bpe8k-t']``,
# which re-runs under the active-vocab mask as part of W1 — no arm here budgets
# a second copy of it.
TREE_TRANSPLANTS: list[TreeTransplantSpec] = [
    TreeTransplantSpec(
        "tree8k-x2",
        "bpe8k",
        budget_ratio=2.0,
        note="the flagship: additive routed table at half the flat floats",
    ),
    TreeTransplantSpec(
        "tree8k-x4",
        "bpe8k",
        budget_ratio=4.0,
        note="how far the sharing goes before it breaks",
    ),
    TreeTransplantSpec(
        "tree8k-x2-film",
        "bpe8k",
        budget_ratio=2.0,
        tree=TreeSpec(film=True),
        note="ablation: FiLM composition, so depth can buy expressivity a "
        "path-sum cannot (stage 0: additive depth-2 == flat VQ)",
    ),
    TreeTransplantSpec(
        "tree8k-x2-fgb",
        "bpe8k",
        budget_ratio=2.0,
        functional_boost=True,
        note="ablation: boost on the loss gradient through the body rather "
        "than on matrix residuals",
    ),
    TreeTransplantSpec(
        "tree-host-x2",
        HOST_ARM,
        budget_ratio=2.0,
        micro_batch=8,
        tree=TreeSpec(n_trees=48),
        note="bridge arm on the active host vocabulary: does training rescue "
        "the distilled init stage 0 showed fails zero-shot?",
    ),
]

TREE_TRANSPLANTS_BY_NAME = {t.name: t for t in TREE_TRANSPLANTS}


# ---------------------------------------------------------------------------
# Study 4, track 4: the progressive-shrink curriculum
# ---------------------------------------------------------------------------
#
# Shrink the *live* pretrained vocabulary by interpretable rules, each followed
# by a healing train, and record BPB against vocabulary size. One-shot
# replacement is what stage 0 killed; a curriculum of small healed edits is what
# the evidence permits. ``shrink.py`` implements the rules; this is the order
# they run in and the only place their hyperparameters live.


@dataclass(frozen=True)
class ShrinkRule:
    """One rung. ``rule`` names a function in ``shrink.RULES``."""

    rule: str  # mask | case_space | repiece_tail
    # repiece_tail: a stem the corpus uses fewer than this many times stops
    # being a row and is spelled out of the stems that survive.
    min_count: int = 0
    # case_space: how many variants a lexeme needs before its group is merged.
    # 2 is not a tuning choice — it is the smallest value that guarantees the
    # canonical row is emitted by something, which is what keeps every row of
    # the shrunk table reachable and lets the softmax mask be dropped.
    min_group: int = 2
    note: str = ""


# Thresholds are measured, not guessed (``python -m ...shrink`` on the real 230 MB
# stream): 100 takes the vocabulary to 17,385 for +1.81% tokens, and 1000 —
# applied *after* it, on re-measured counts — to 4,783 for +14.07% against the
# host's own stream. Adding a rung is one line, so if the curve needs filling
# between them, fill it.
SHRINK_CURRICULUM: tuple[ShrinkRule, ...] = (
    ShrinkRule("mask", note="rule 0 — W0's active-vocab mask, as deletion"),
    ShrinkRule(
        "case_space",
        note="rule 1 — case/space variants of one lexeme become stem + role "
        "bit; a relabeling, so the token count does not move",
    ),
    ShrinkRule("repiece_tail", min_count=100, note="rule 2 — re-piece the tail"),
    ShrinkRule("repiece_tail", min_count=1000, note="rule 2 again, deeper"),
)


# Rule 1 merges token *ids*, and measurement says that is not the same thing as
# merging case. `shrink1-case` is 0.0039 BPB better than native on clean text and
# +1.145 worse on `upper`, with 0.886 of the damage in the stem head — because
# the host tokenizer fragments uppercase text into different pieces *before* any
# merge applies (`orig-frozen` emits +64% tokens on `upper`), so the merge fires
# on the wrong fragments. Diagnostic, 2026-08-30: the stem streams rule 1
# produces for `upper` and `lower` share only 36.9% of their unigram mass.
#
# This curriculum swaps the merge for a fold at tokenization time: tokenize the
# lowercased text, emit the role from the original casing. Same vocabulary, same
# role algebra; the difference is entirely in how the stream is produced, so the
# `case_space` arms keep the numbers already published against them.
SHRINK_FOLD_CURRICULUM: tuple[ShrinkRule, ...] = (
    ShrinkRule("mask", note="rule 0 — unchanged; the active set is the folded one"),
    ShrinkRule(
        "case_fold",
        min_group=1,
        note="rule 1' — the merged table entered by re-tokenizing the "
        "lowercased text; min_group=1 is what makes the encoder's decision a "
        "function of a surface rather than of the plan",
    ),
    ShrinkRule("repiece_tail", min_count=100, note="rule 2 — re-piece the tail"),
    ShrinkRule("repiece_tail", min_count=1000, note="rule 2 again, deeper"),
)

SHRINK_CURRICULA: dict[str, tuple[ShrinkRule, ...]] = {
    "default": SHRINK_CURRICULUM,
    "fold": SHRINK_FOLD_CURRICULUM,
}


@dataclass(frozen=True)
class ShrinkTransplantSpec:
    """One point on the BPB-vs-vocabulary-size curve.

    ``stage`` is how many rules of the arm's curriculum to apply, so the arms
    are nested vocabularies and the curriculum's order is the grid.

    Execution fields carry the same names and meanings as ``TransplantSpec`` so
    ``train_transplant.run`` drives either kind without branching on type; a
    shrink arm differs only by having a ``stage``, which is what the re-indexing
    hook keys on.
    """

    name: str
    stage: int
    # Replaces the *last* applied rule's ``min_count``. The curriculum is a
    # linear chain and ``stage`` a prefix of it, so this is how a rung gets a
    # different depth without forking the chain: rungs 2 and 3 differ only in
    # where the re-piecing threshold sits, and the curve needs points between
    # them far more than it needs a longer chain.
    override_min_count: int = 0
    # Which chain of ``SHRINK_CURRICULA`` ``stage`` is a prefix of.
    curriculum: str = "default"
    arm: str = HOST_ARM
    init: str = "shrink"
    train: bool = True
    micro_batch: int = 8
    freeze_body: bool = True
    note: str = ""


# Each rung is healed from the *host* table through its own codec rather than
# warm-started from the previous rung's checkpoint. That is a deliberate
# deviation from the pre-registered wording ("each followed by a short healing
# train", which reads as sequential): with sequential healing, rung k would
# carry k times rung 1's training compute, and the curve would confound
# vocabulary size with optimizer steps — precisely the confound the fixed-bytes
# protocol exists to remove. Equal budget per rung also makes the arms
# independent, so they launch as parallel jobs.
SHRINKS: list[ShrinkTransplantSpec] = [
    ShrinkTransplantSpec(
        "shrink0-mask",
        stage=1,
        train=False,
        note="rule 0 only, eval only — must reproduce orig-frozen's 0.7796, "
        "which is the machinery's correctness check rather than a measurement",
    ),
    ShrinkTransplantSpec(
        "shrink1-case",
        stage=2,
        note="the first rung where parameters are actually deleted: 74,957 -> "
        "48,703 stems plus a 6-value role head",
    ),
    ShrinkTransplantSpec(
        "shrink2-tail100",
        stage=3,
        note="-> 17,385 stems (35.6M floats) at +1.81% tokens",
    ),
    ShrinkTransplantSpec(
        "shrink3-tail1k",
        stage=4,
        note="-> 4,783 stems (9.8M floats, x31.7 against the host table) at "
        "+14.07% tokens; where the gate is expected to bite, and the point of "
        "the curve is to say where it actually does",
    ),
    # The gate flips between stage-3 thresholds 100 (PASS, -0.0021) and 1000
    # (FAIL, +0.0348), a 3.6x jump in vocabulary with nothing measured inside
    # it. The deliverable of this track is a *curve*; these are the points
    # where it actually breaks.
    ShrinkTransplantSpec(
        "shrink2-tail200",
        stage=3,
        override_min_count=200,
        note="locating the cliff between x4.3 (passes) and x15.6 (fails)",
    ),
    ShrinkTransplantSpec(
        "shrink2-tail300", stage=3, override_min_count=300, note="ditto"
    ),
    ShrinkTransplantSpec(
        "shrink2-tail500", stage=3, override_min_count=500, note="ditto"
    ),
    # --- the case-fold repair (W2 follow-up) --------------------------------
    # These are *additions*, not replacements: `shrink1-case` and the tail arms
    # keep the numbers already on the curve, and the comparison that matters is
    # fold-vs-merge at the same vocabulary size and the same token budget.
    ShrinkTransplantSpec(
        "shrink1-fold",
        stage=2,
        curriculum="fold",
        note="rule 1's table, entered by folding case at tokenization time — "
        "the arm that tests whether shrink1-case's +1.145 on `upper` was the "
        "merge or the fragmentation the merge inherited",
    ),
    ShrinkTransplantSpec(
        "shrink2-fold100",
        stage=3,
        curriculum="fold",
        note="the fold carried through one re-piecing pass, so the compression "
        "that passed the gate (x4.31) is measured on the case-invariant stream "
        "rather than only at the free rung",
    ),
]

SHRINKS_BY_NAME = {s.name: s for s in SHRINKS}


# ---------------------------------------------------------------------------
# Study 4, track 3: calibrating the fitness ladder before evolving (ADR 0002)
# ---------------------------------------------------------------------------
#
# The candidate set is where this track lives or dies. Correlating a cheap rung
# against an expensive one over near-identical candidates inflates rho — the
# same range-restriction trap this study already documented once, when
# `inflation` rewarded a bad baseline. So the set is feature-group ablations
# PLUS deliberately-bad anchors, and each anchor is a *different kind* of bad:
# `random` has no signal, `length` has real but nearly useless signal, and the
# single-feature routers have signal on exactly one axis.


@dataclass(frozen=True)
class CalibrationCandidate:
    name: str
    kind: str  # groups | columns | random
    groups: tuple[str, ...] = ()
    columns: tuple[str, ...] = ()
    is_anchor: bool = False
    note: str = ""


CALIBRATION_CANDIDATES: list[CalibrationCandidate] = [
    # --- feature-group ablations: the range we actually care about ----------
    CalibrationCandidate("all", "groups", groups=FEATURE_GROUPS_V0),
    CalibrationCandidate("surface", "groups", groups=("surface",)),
    CalibrationCandidate("ngram", "groups", groups=("ngram",)),
    CalibrationCandidate("corpus", "groups", groups=("corpus",)),
    CalibrationCandidate("decomposition", "groups", groups=("decomposition",)),
    CalibrationCandidate("surface+ngram", "groups", groups=("surface", "ngram")),
    CalibrationCandidate("surface+corpus", "groups", groups=("surface", "corpus")),
    CalibrationCandidate(
        "surface+decomposition", "groups", groups=("surface", "decomposition")
    ),
    CalibrationCandidate("ngram+corpus", "groups", groups=("ngram", "corpus")),
    CalibrationCandidate(
        "no-ngram", "groups", groups=("surface", "decomposition", "corpus")
    ),
    CalibrationCandidate(
        "no-corpus", "groups", groups=("surface", "decomposition", "ngram")
    ),
    CalibrationCandidate(
        "no-surface", "groups", groups=("decomposition", "ngram", "corpus")
    ),
    # --- deliberately-bad anchors: the low end rho needs to be honest -------
    CalibrationCandidate(
        "anchor-random",
        "random",
        is_anchor=True,
        note="no signal at all — the floor any usable rung must rank last",
    ),
    CalibrationCandidate(
        "anchor-length",
        "columns",
        columns=("len.bytes.log",),
        is_anchor=True,
        note="length-only: real signal, nearly useless for sharing parameters",
    ),
    CalibrationCandidate(
        "anchor-logfreq",
        "columns",
        columns=("corpus.logfreq",),
        is_anchor=True,
        note="single-feature: frequency alone, the most plausible bad router",
    ),
    CalibrationCandidate(
        "anchor-leading-space",
        "columns",
        columns=("flag.leading_space",),
        is_anchor=True,
        note="one bit: at most two groups of tokens",
    ),
]

CALIBRATION_CANDIDATES_BY_NAME = {c.name: c for c in CALIBRATION_CANDIDATES}


# ---------------------------------------------------------------------------
# Study 4, stage 0: distill the host's embedding table into structured forms
# ---------------------------------------------------------------------------
#
# The load-bearing question for the tree-vocabulary program: how many float
# parameters does the pretrained embedding *function* actually need? Every
# method fits the frozen table offline (no LM training anywhere); the
# reconstruction is plugged into the frozen body and evaluated with the
# study-2 BPB harness. Kill criterion agreed up front: > 0.05 BPB damage at
# ~10x compression kills the program; the hope is < 0.01 at ~20x.


@dataclass(frozen=True)
class DistillSpec:
    name: str
    method: str  # svd | vq | rvq | tree | tree_res
    rank: int = 0  # svd: rank; tree_res: residual rank
    codes: int = 0  # vq/rvq: codebook size; tree/tree_res: branching factor
    levels: int = 0  # rvq: number of residual codebook levels
    depth: int = 0  # tree/tree_res: depth
    weight: str = "none"  # none | freq -- row weights for the fit objective
    apply: str = "both"  # both | input | output -- which side is reconstructed
    # Rescale reconstructed rows to the native per-row norms (V extra floats).
    # Controls for the softmax-temperature artifact: low-rank fits shrink row
    # norms, which damages logit scale independently of direction error.
    renorm: bool = False
    full_suites: bool = False  # evaluate all perturbation suites, not just clean
    note: str = ""


# The ladders are matched in float-parameter budget (~x4 / x8 / x16 / x32
# against the 311.2M-float native table) so methods compare at equal cost.
# svd shares one exact SVD across ranks; rvq levels share one fit; the tree
# arms share one tree and one residual basis. Adding a rung is nearly free.
DISTILLS: list[DistillSpec] = [
    # --- linear floor: global low-rank -------------------------------------
    DistillSpec("svd512", "svd", rank=512, note="~x4.0"),
    DistillSpec("svd256", "svd", rank=256, note="~x7.9"),
    DistillSpec("svd128", "svd", rank=128, full_suites=True, note="~x15.8"),
    DistillSpec("svd64", "svd", rank=64, note="~x31.6"),
    # --- discrete-only floor: every token snaps to a centroid --------------
    DistillSpec("vq4k", "vq", codes=4096, note="~x37, flat codebook"),
    DistillSpec("tree64x2", "tree", codes=64, depth=2, note="~x37, routed"),
    # --- additive codes, unrouted ------------------------------------------
    DistillSpec("rvq8x4k", "rvq", codes=4096, levels=8, note="~x4.6"),
    DistillSpec("rvq4x4k", "rvq", codes=4096, levels=4, note="~x9.3"),
    DistillSpec(
        "rvq2x4k", "rvq", codes=4096, levels=2, full_suites=True, note="~x18.5"
    ),
    # --- the proposal: tree neighborhood + low-rank token identity ---------
    DistillSpec("tree64x2-r448", "tree_res", codes=64, depth=2, rank=448, note="~x4.0"),
    DistillSpec("tree64x2-r192", "tree_res", codes=64, depth=2, rank=192, note="~x8.2"),
    DistillSpec(
        "tree64x2-r64",
        "tree_res",
        codes=64,
        depth=2,
        rank=64,
        full_suites=True,
        note="~x16.9",
    ),
    DistillSpec("tree64x2-r8", "tree_res", codes=64, depth=2, rank=8, note="~x31.9"),
    # --- does the fit objective matter? weight rows by token frequency -----
    DistillSpec("svd128-freq", "svd", rank=128, weight="freq", full_suites=True),
    DistillSpec(
        "tree64x2-r64-freq",
        "tree_res",
        codes=64,
        depth=2,
        rank=64,
        weight="freq",
        full_suites=True,
    ),
    # --- which side of the model is sensitive? -----------------------------
    DistillSpec("svd128-in", "svd", rank=128, apply="input", note="exact output head"),
    DistillSpec("svd128-out", "svd", rank=128, apply="output", note="exact input emb"),
    # --- softmax-temperature control: native row norms restored ------------
    DistillSpec(
        "svd128-freq-norm",
        "svd",
        rank=128,
        weight="freq",
        renorm=True,
        note="rules out row-norm shrinkage as the damage mechanism",
    ),
    DistillSpec(
        "tree64x2-r64-freq-norm",
        "tree_res",
        codes=64,
        depth=2,
        rank=64,
        weight="freq",
        renorm=True,
    ),
]

DISTILLS_BY_NAME = {s.name: s for s in DISTILLS}
