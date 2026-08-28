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
]

TRANSPLANTS_BY_NAME = {t.name: t for t in TRANSPLANTS}


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
