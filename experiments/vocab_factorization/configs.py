"""Experiment grid, as code. This module is the single source of truth.

Two levels:

``ARMS``        tokenizer arms -- what gets built and screened without training
``EXPERIMENTS`` training runs  -- one small LM per row of the ablation grid

Nothing about the grid lives in the plan markdown; the markdown points here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
]

ARMS_BY_NAME = {a.name: a for a in ARMS}

# Bytes of training text used to fit the tokenizers. 50 MB is far more than a
# 32k vocabulary needs and keeps arm construction to a couple of minutes.
TOKENIZER_TRAIN_BYTES = 50 * 1024 * 1024


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
]

EXPERIMENTS_BY_NAME = {e.name: e for e in EXPERIMENTS}
