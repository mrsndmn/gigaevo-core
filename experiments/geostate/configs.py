"""GeoState decisive probe — the grid, as code.

Single source of truth for the symmetry group, generator tolerances, the E0
consistency arms and the kill gates. The plan (`experiments/geostate.md`) and
the interview spec point here and repeat nothing.

Like the other `experiments/*/configs.py`, this file imports **nothing but the
standard library**, so a launcher running under a different conda env (e.g.
`arkhip`, which has `mls` but no `numpy`) can still read the grid.

Decisions recorded as ADRs: placement (0004), symmetry group (0005), matched
baseline and gates (0006).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "run" / "geostate"  # local runs and smoke tests
# MLS jobs write outside the staged workdir, on the shared NFS artifacts root
# (`run_e0_jobs.py` rewrites it to the worker's mount).
ARTIFACT_ROOT = "/mnt/virtual_ai0001053-00054_SR004-nfs2/d.tarasov/gigaevo-artifacts/results/geostate"

# Same endpoint as noregeo_synth; credentials are resolved at runtime by
# `experiments.noregeo_synth.credentials`.
DEFAULT_BASE_URL = "https://inference.airi.net/v1"

# The endpoint omits its intermediate certificate, so the default CA store
# fails with CERTIFICATE_VERIFY_FAILED. The public, CA-issued GlobalSign
# intermediate is appended to certifi's roots at runtime (`client.ca_bundle`),
# keeping normal root and hostname validation. Provenance:
# https://secure.globalsign.com/cacert/gsgccr3dvtlsca2020.crt (same file as
# noregeo_synth/certs, copied so the package stays self-contained).
INTERMEDIATE_CERT = (
    Path(__file__).resolve().parent / "certs" / "globalsign-gcc-r3-dv-tls-ca-2020.pem"
)
TLS_BUNDLE_NAME = "tls-ca-bundle.pem"

# ---------------------------------------------------------------------------
# Symmetry group (ADR 0005)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroupSpec:
    """Orientation-preserving similarities: translation, rotation, uniform scale.

    Reflection is *not* part of the group; `reflection=True` exists only for
    the out-of-distribution evaluation condition.
    """

    translation_max: float = 6.0  # each component uniform in [-max, max]
    scale_min: float = 0.5
    scale_max: float = 2.0
    reflection: bool = False


GROUP = GroupSpec()
GROUP_OOD_REFLECTION = GroupSpec(reflection=True)

# Kinds of copies E0 makes of one construction problem. Index 0 of every
# problem is always `identity`; the others cycle through the rest so that the
# report can break consistency down by *which* transform breaks a model.
TRANSFORM_KINDS: tuple[str, ...] = (
    "identity",
    "relabel",
    "translate",
    "rotate",
    "scale",
    "similarity",
    "similarity+relabel",
)

# Out-of-distribution condition (ADR 0005): a reflection composed with a
# rotation, never part of the training group. Copies whose target is not the
# reflection of the original target (orientation-dependent constructions such
# as a quarter turn) are rejected at preparation time, so the OOD set only
# contains problems where the question is well posed.
OOD_KINDS: tuple[str, ...] = ("identity", "reflect", "reflect+relabel")

# ---------------------------------------------------------------------------
# Kernel tolerances — all relative to the state's characteristic length (the
# diameter of the given points), so every predicate is invariant under scaling.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToleranceSpec:
    # Predicates. The model's program executes in the same exact kernel as the
    # target, so these only need to absorb float error.
    coincidence: float = 1e-6
    incidence: float = 1e-6
    tangency: float = 1e-6
    # Degeneracy: moves whose result is closer than this to degenerate are
    # rejected (spec risk 5). Applies to generated and to model-written moves
    # alike, so the environment has one notion of "valid move".
    min_point_separation: float = 0.02  # line(P, Q), circle(O, P)
    min_meet_separation: float = 0.02  # two intersection points (near-tangent)
    min_line_angle_deg: float = 2.0  # line-line meet (near-parallel)
    # Generator-only sanity: constructed points must stay within this multiple
    # of the diameter from the given points' centroid.
    max_extent: float = 8.0
    # Cross-copy agreement in E0: given coordinates are rounded to 2 decimals
    # after a transform, so answers mapped back to the original frame agree
    # only up to this relative distance.
    agreement: float = 2e-2


TOL = ToleranceSpec()

# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

LENGTH_BUCKETS: tuple[tuple[str, int, int | None], ...] = (
    ("1-3", 1, 3),
    ("4-9", 4, 9),
    ("10+", 10, None),
)


def bucket_of(n_moves: int) -> str:
    for name, lo, hi in LENGTH_BUCKETS:
        if n_moves >= lo and (hi is None or n_moves <= hi):
            return name
    raise ValueError(f"no bucket for {n_moves} moves")


@dataclass(frozen=True)
class GeneratorSpec:
    coord_min: int = -10  # given points on the NoReGeo integer grid
    coord_max: int = 10
    n_given_min: int = 3
    n_given_max: int = 5
    max_depth: int = 2  # nesting of point-valued macros
    max_macros: int = 3  # macro nodes per expression, so descriptions stay readable
    compose_prob: float = 0.45  # chance a point slot is a nested macro
    max_moves: int = 20  # the 10+ bucket is 10..max_moves
    max_attempts_per_problem: int = 50
    labels: str = "ABCDEFGH"
    # Goals with several valid answers (AC-10): drawn with this probability
    # (60% incidence, 40% tangency), and capped per length bucket so that no
    # bucket is dominated by one goal kind. A tangent through a point needs
    # ten primitive moves (Thales), so it only ever lands in the long bucket.
    predicate_goal_prob: float = 0.45  # draw rate; the per-bucket caps below bind
    incident_share: float = 0.25  # of a bucket, at most
    tangent_share: float = 0.15  # of a bucket, at most


GENERATOR = GeneratorSpec()

# ---------------------------------------------------------------------------
# Auxiliary construction set: NoReGeo stems whose golden answer is a point a
# construction produces. stem -> macro. Reflection and rotation need extra
# given points (the mirror axis / the centre), added by `aux.py`.
# ---------------------------------------------------------------------------

AUX_STEMS: dict[str, str] = {
    "midpoint": "midpoint",
    "intersection": "line_meet",
    "bisector": "angle_bisector_foot",
    "inner_circle_center": "incenter",
    "outer_circle_center": "circumcenter",
    "reflection": "reflect",
    "rotation_point": "rotate",
}

# ---------------------------------------------------------------------------
# Kill gates (ADR 0006), fixed before anything runs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class E0Gate:
    min_moves: int = 5  # the bucket the gate is read on: problems with >= this
    consistency_kill: float = 0.95  # best open model at/above this => H0 dead
    # The verdict is withheld unless the data can carry it: every copy must
    # have been evaluated (no request errors), most copies must have produced
    # a parseable answer, and the model must solve a fair share of the
    # untransformed problems — a model that cannot use the language, or a run
    # with an expired token, is "consistent" for the wrong reason.
    min_answered_rate: float = 0.9
    min_success_identity: float = 0.5


@dataclass(frozen=True)
class V1Gate:
    min_moves: int = 5
    n_eval: int = 300
    go_delta_pts: float = 10.0  # success-rate gain, with bootstrap 95% CI > 0
    kill_delta_pts: float = 3.0  # at/below this => no-go; between => reconsider


E0_GATE = E0Gate()
V1_GATE = V1Gate()

# ---------------------------------------------------------------------------
# E0 consistency arms
# ---------------------------------------------------------------------------

ARM_TEXT = "text"  # coordinates in the prompt, no image
ARM_FIGURE = "figure"  # labelled points rendered on a grid, no coordinates
ARMS: tuple[str, ...] = (ARM_TEXT, ARM_FIGURE)

SET_SYNTHETIC = "synthetic"
SET_AUX = "aux"

# The roster the endpoint serves changes without notice: on 2026-09-24 the
# smoke run watched Qwen3.8-27B disappear mid-run, and Qwen3.5-27B / Qwen3-Next
# were already gone; what remained was Qwen3-8B, Gpt-oss-120b (open-weight,
# text-only) and the two Medgemmas. `consistency.run` lists the served models
# at start and skips absent ones, so a config may name a model that is not
# there today and pick it up on a later run.
MODEL_QWEN38 = "Qwen/Qwen3.8-27B"  # figure-reading VLM; served again 2026-09-30
MODEL_QWEN3_8B = "Qwen/Qwen3-8B"
MODEL_GPTOSS = "Openai/Gpt-oss-120b"  # text-only: a blind control on the figure arm
MODEL_GEMMA4 = "Google/Gemma-4-31B-it"  # open-weight, first served 2026-09-30
MODEL_QWEN35 = "Qwen/Qwen3.5-27B"  # absent since 2026-09-24
MODEL_QWEN3_NEXT = "Qwen/Qwen3-Next-80B-A3B-Instruct"  # absent since 2026-09-24
# Google/Medgemma-27b-it is excluded from the figure arm: `noregeo-vlm-eval.md`
# established that it cannot read the rendered figure.
# Order is the request order: `consistency.run` queues every copy of the first
# model before the second. The H0 gate reads the best open model, so the
# likely strongest go first and Qwen3-8B (16k-token thinking, ~6x slower) goes
# last — in the 2026-09-24 runs it went first and the token expired before any
# other model was reached.
TEXT_MODELS: tuple[str, ...] = (
    MODEL_QWEN38,
    MODEL_GPTOSS,
    MODEL_GEMMA4,
    MODEL_QWEN3_8B,
)


@dataclass(frozen=True)
class E0Config:
    name: str
    models: tuple[str, ...]
    arm: str
    problem_set: str
    copies: int = 8  # per problem, index 0 = identity
    per_bucket: int = 100  # synthetic problems per length bucket
    per_stem: int = 20  # auxiliary problems per stem
    kinds: tuple[str, ...] = TRANSFORM_KINDS
    group: GroupSpec = GROUP
    seed: int = 20260924
    base_url: str = DEFAULT_BASE_URL
    temperature: float = 0.0
    top_p: float = 1.0
    # Thinking models (Qwen3-8B) spend 2.5-5k tokens on a 3-move problem and
    # blew a 6144 cap in the smoke; 16k leaves room without the 32k context.
    max_tokens: int = 16384
    # vLLM reasoning control, passed through `extra_body`; None = omit.
    reasoning_effort: str | None = None
    # Qwen3-8B decodes at ~43 tok/s: a 16k completion needs ~380 s.
    request_timeout: float = 900.0
    max_transport_retries: int = 6
    retry_backoff_cap: float = 60.0
    workers: int = 4  # the live smoke ran 4 in flight without a 429
    bootstrap: int = 1000
    # MLS instance for the inference-only job; the allocation is GPU-only and
    # this is the type noregeo's inference jobs use.
    instance_type: str = "a100.1gpu"
    # Only a gated config's summary prints the H0 verdict (AC-14): the figure
    # arm and the auxiliary set are reported, not judged.
    gated: bool = False


# ---------------------------------------------------------------------------
# V1: state tokens vs matched-training baseline (ADR 0006, AC-15..19)
# ---------------------------------------------------------------------------

ARM_TOKENS = "tokens"  # state blocks carry one <st> token per object (encoder)
ARM_TABLE = "table"  # state blocks carry the object's parameters as text


@dataclass(frozen=True)
class V1Config:
    name: str = "v1"
    base_model: str = "Qwen/Qwen3-1.7B"  # in the shared HF cache (HF_HOME)
    arms: tuple[str, ...] = (ARM_TOKENS, ARM_TABLE)
    seeds: tuple[int, ...] = (1, 2)
    # data: generator trajectories, held out by seed (AC-15)
    train_seed: int = 1000
    train_per_bucket: int = 16667  # ~50k trajectories over the three buckets
    eval_seed: int = 2000
    eval_per_bucket: int = 300
    max_len: int = 1536  # longer teacher-forced sequences are dropped
    # invariant-feature encoder
    d_enc: int = 256
    enc_layers: int = 2
    # LoRA, identical in both arms (AC-16)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    # optimisation
    lr_lora: float = 2e-4
    lr_encoder: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 1
    # Effective batch 16. Full-vocabulary logits over 1.5k-token sequences
    # are the memory hog: 8 sequences per micro-batch left no room for a
    # second process on an 80 GB card in the smoke, 4 do.
    batch_size: int = 4
    grad_accum: int = 4
    gradient_checkpointing: bool = True
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    log_every: int = 25
    # rollout: move budget relative to the reference trajectory
    move_budget_factor: float = 2.0
    move_budget_extra: int = 4
    move_budget_cap: int = 30
    max_move_tokens: int = 24  # tokens per generated move line
    # MLS
    instance_type: str = "a100.1gpu"
    # local smoke
    smoke_train_per_bucket: int = 12
    smoke_steps: int = 20
    smoke_eval_per_bucket: int = 2


V1 = V1Config()

# ---------------------------------------------------------------------------
# H4: the trained V1 adapters under poses outside the training range
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PoseCondition:
    """One family of poses applied to every frozen V1 evaluation problem.

    The map is x -> scale * R(theta) (F x) + t: theta uniform when `rotate`,
    F the x-axis reflection when `reflect`, t of length `translation` in a
    uniform direction. Given coordinates are then rounded to two decimals and
    the reference re-derived, exactly as E0 builds its copies.
    """

    name: str
    rotate: bool = False
    scale: float = 1.0
    translation: float = 0.0
    reflect: bool = False
    # Given coordinates outside the training range (integer grid in
    # [-10, 10]): the conditions the H4 gate is read on.
    far: bool = False


POSE_CONDITIONS: tuple[PoseCondition, ...] = (
    # control: must reproduce the saved V1 episodes (adapter loading check)
    PoseCondition("identity"),
    # in the group, near the training range; coordinates stop being integers
    PoseCondition("rotate", rotate=True),
    PoseCondition("translate-far", translation=100.0, far=True),  # ~10x the box
    PoseCondition("scale-up", scale=10.0, far=True),
    # Reported, not gated: the given points stay inside the box, only the
    # figure shrinks — and the table prints at a fixed two decimals, so a drop
    # here partly measures print precision, not pose.
    PoseCondition("scale-down", scale=0.2),
    # ADR 0005's out-of-distribution condition: orientation flips, so the
    # tokens' signed features change too — neither arm is invariant here
    PoseCondition("reflect", rotate=True, reflect=True),
)


@dataclass(frozen=True)
class H4Gate:
    """Robustness on the long bucket, per far condition, paired per problem.

    The statistic is (tokens − tokens at identity) − (table − table at
    identity): how much less the tokens arm drops than the table arm when the
    pose leaves the training range. Not the absolute tokens − table gap — at
    identity the table already leads by ~19 pts (H1), so a gate on the gap
    would read KILL even if the table collapsed and the tokens did not move.
    GO if any far condition shows at least `go_delta_pts` with its bootstrap
    CI above zero; KILL if every far condition is at/below `kill_delta_pts`;
    RECONSIDER otherwise. Same margins as H1. Withheld unless every run is
    complete and identity reproduces the saved V1 episodes. Fixed before any
    out-of-distribution episode exists (2026-09-30).
    """

    min_moves: int = 5
    n_min: int = 300  # paired long problems per far condition, as H1's n_eval
    go_delta_pts: float = 10.0
    kill_delta_pts: float = 3.0
    # identity episodes whose success matches the saved V1 episode, per run
    min_reproduction: float = 0.95


H4_GATE = H4Gate()


@dataclass(frozen=True)
class V1OODConfig:
    name: str = "v1-ood"
    # The runs whose adapters are evaluated — V1 protocol B, the one H1 read —
    # and the frozen evaluation set they were rolled out on.
    source_protocol: str = "89abc7561426"
    data_key: str = "0493e3e860"
    conditions: tuple[PoseCondition, ...] = POSE_CONDITIONS
    seed: int = 4000  # poses; per problem and condition, independent of order
    max_attempts: int = 20  # re-draws of a random pose a problem rejects
    smoke_per_bucket: int = 2
    instance_type: str = "a100.1gpu"


V1_OOD = V1OODConfig()

experiment_configs: list[E0Config] = [
    E0Config(
        name="e0-synthetic-text",
        models=TEXT_MODELS,
        arm=ARM_TEXT,
        problem_set=SET_SYNTHETIC,
        gated=True,
    ),
    E0Config(
        name="e0-aux-text",
        models=TEXT_MODELS,
        arm=ARM_TEXT,
        problem_set=SET_AUX,
        per_stem=100,  # every valid row of the seven stems
    ),
    E0Config(
        name="e0-synthetic-text-reflect",
        models=(MODEL_QWEN38,),
        arm=ARM_TEXT,
        problem_set=SET_SYNTHETIC,
        copies=4,
        kinds=OOD_KINDS,
        group=GROUP_OOD_REFLECTION,
    ),
    E0Config(
        name="e0-aux-figure",
        models=(MODEL_QWEN38,),
        arm=ARM_FIGURE,
        problem_set=SET_AUX,
    ),
    E0Config(
        name="e0-synthetic-figure",
        models=(MODEL_QWEN38,),
        arm=ARM_FIGURE,
        problem_set=SET_SYNTHETIC,
    ),
]

DEFAULT_CONFIG = experiment_configs[0].name


def config_named(name: str) -> E0Config:
    for cfg in experiment_configs:
        if cfg.name == name:
            return cfg
    known = ", ".join(c.name for c in experiment_configs)
    raise KeyError(f"unknown config {name!r}; known: {known}")
