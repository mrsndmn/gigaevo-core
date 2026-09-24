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
MODEL_QWEN38 = "Qwen/Qwen3.8-27B"  # the only figure-reading VLM; absent now
MODEL_QWEN3_8B = "Qwen/Qwen3-8B"
MODEL_GPTOSS = "Openai/Gpt-oss-120b"  # text-only: a blind control on the figure arm
MODEL_QWEN35 = "Qwen/Qwen3.5-27B"  # absent since 2026-09-24
MODEL_QWEN3_NEXT = "Qwen/Qwen3-Next-80B-A3B-Instruct"  # absent since 2026-09-24
# Google/Medgemma-27b-it is excluded from the figure arm: `noregeo-vlm-eval.md`
# established that it cannot read the rendered figure.
TEXT_MODELS: tuple[str, ...] = (MODEL_QWEN3_8B, MODEL_GPTOSS, MODEL_QWEN38)


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
