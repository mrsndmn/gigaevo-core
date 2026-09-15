"""Experiment grid for the NoReGeo reasoning+code synthesis, as code.

This module is the single source of truth for *what* is generated and with
which knobs; the plan markdown (`experiments/noregeo-synth.md`) points here and
repeats nothing.

Like `experiments/vocab_factorization/configs.py`, this file imports **nothing
but the standard library** so that a launcher running under a different conda
env (e.g. `arkhip`, which has `mls` but no `matplotlib`) can still read the
grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Source benchmark
# ---------------------------------------------------------------------------

HF_REPO = "FusionBrainLab/NoReGeo"
HF_REVISION = "main"
HF_RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}"

# The 27 task files in the repo (100 tasks each). The HF auto-converted parquet
# ("default/train") holds only images, so the question/answer text has to come
# from these JSONL files.
TASK_FILES: tuple[str, ...] = (
    "area_comparison/parallelogram_size/parallelogram_size_tasks.jsonl",
    "area_comparison/triangle_size/triangle_size_tasks.jsonl",
    "basic_coordinate/collinearity/collinearity_tasks.jsonl",
    "basic_coordinate/midpoint_coordinates/midpoint_tasks.jsonl",
    "basic_coordinate/parallelogram_area/parallelogram_area_tasks.jsonl",
    "basic_coordinate/parallelogram_perimeter/parallelogram_perimeter_tasks.jsonl",
    "basic_coordinate/triangle_area/triangle_area_tasks.jsonl",
    "basic_coordinate/triangle_perimeter/triangle_perimeter_tasks.jsonl",
    "basic_coordinate/triangle_type/triangle_type_tasks.jsonl",
    "circle_properties/diameter/circle_diameter_tasks.jsonl",
    "circle_properties/inner_circle/inner_circle_center_tasks.jsonl",
    "circle_properties/inner_circle_radius/inner_circle_radius_tasks.jsonl",
    "circle_properties/outer_circle/outer_circle_center_tasks.jsonl",
    "circle_properties/outer_circle_radius/outer_circle_radius_tasks.jsonl",
    "circle_properties/semicircle_triangle/semicircle_triangle_tasks.jsonl",
    "elementary_calculations/center_coordinates/center_coordinates_tasks.jsonl",
    "elementary_calculations/intersection_segment/intersection_tasks.jsonl",
    "elementary_calculations/segment_length/segment_length_tasks.jsonl",
    "geom_transformation/reflection/reflection_tasks.jsonl",
    "geom_transformation/rotation/rotation_point_tasks.jsonl",
    "parallelism/parallel_lines_tasks.jsonl",
    "parallelism_and_perpendicularity/perpendicular/perpendicular_tasks.jsonl",
    "parallelism_and_perpendicularity/right_angle/right_angle_tasks.jsonl",
    "remarkable_lines/bisector/bisector_tasks.jsonl",
    "remarkable_lines/special_lines/special_lines_tasks.jsonl",
    "symmetry/shape_symmetry/shape_symmetry_tasks.jsonl",
    "symmetry/symmetry_points/symmetry_tasks.jsonl",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "noregeo"  # cached benchmark JSONL + images
OUT_DIR = REPO_ROOT / "run" / "noregeo_synth"  # generated dataset + figures

# ---------------------------------------------------------------------------
# Inference endpoint
# ---------------------------------------------------------------------------

# Internal OpenAI-compatible vLLM gateway. Credentials are resolved at runtime
# by `credentials.py` (env vars first, then a small list of env files) so no
# token ever lands in the repo.
DEFAULT_BASE_URL = "https://inference.airi.net/v1"

# Models served there as of 2026-09-09 (`GET /v1/models`), with context length:
#   AIRI/OCC-1-1.7B                   32k
#   Google/Medgemma-1.5-4b-it         64k     (medical)
#   Google/Medgemma-27b-it            32k     (medical)
#   Openai/Gpt-oss-120b              131k
#   Qwen/Qwen3-8B                     32k
#   Qwen/Qwen3-Next-80B-A3B-Instruct 262k     (80B MoE, 3B active)
#   Qwen/Qwen3.5-27B                 131k
#   Qwen/Qwen3.8-27B                 262k     <- newest small dense, default
#   BAAI/bge-m3                        8k     (embeddings)


# Concurrent samples in flight during a batch run. The endpoint is a shared
# vLLM server that batches requests, so throughput keeps climbing well past the
# point where a single request slows down; the ceiling is politeness towards
# other users of the server, not our own efficiency. Measured with
# `python -m experiments.noregeo_synth.probe_concurrency` — see the plan doc.
DEFAULT_WORKERS = 8


@dataclass(frozen=True)
class GenConfig:
    """One generation arm: a model plus the knobs that shape the output."""

    name: str
    model: str = "Qwen/Qwen3.8-27B"
    base_url: str = DEFAULT_BASE_URL

    # sampling
    temperature: float = 0.3
    top_p: float = 0.9
    # A well-formed solution costs ~1.8k completion tokens (3.6k for the worst
    # of 27 measured). The cap is what a *degenerate* generation costs before
    # it gives up, so keeping it tight is the difference between a 1-minute and
    # a 5-minute retry: the two stragglers in the 27-sample pilot took 683s and
    # 398s, against a 53s median.
    max_tokens: int = 6144
    # Where a retry may escalate to when the completion was cut off because the
    # solution genuinely did not fit, rather than because it degenerated.
    max_tokens_ceiling: int = 16384
    seed: int | None = 1234  # vLLM honours this -> reproducible generations
    # A healthy call takes ~50-100 s and a degenerate one ends when it hits
    # max_tokens, so 300 s is generous. It matters because a stalled call holds
    # a worker (1/8 of a batch run's throughput) for its whole duration.
    request_timeout: float = 300.0

    # Hidden chain-of-thought:
    #   "off"                 -> chat_template_kwargs={"enable_thinking": False}
    #   "low"/"medium"/"high" -> reasoning_effort=<level>
    #   None                  -> whatever the server defaults to
    #
    # "low" is not a compromise, it is the only setting that works here
    # (measured 2026-09-09 on Qwen3.8-27B, same prompt, seed 1234):
    #   default (full thinking): burns all 12288 tokens on thought, emits no
    #       content at all (finish_reason="length");
    #   "off": valid grammar, but the sampler degenerates into whitespace right
    #       after the first JSON field — 107k chars of "\n" and a truncated,
    #       unparseable object. A frequency_penalty of 0.3 papers over it
    #       (23k chars, 828 newline runs) but pollutes the code formatting;
    #   "low": 1818 completion tokens, clean JSON, 4 well-formed steps, ~45s.
    thinking: str | None = "low"

    # Retries for a completion that comes back unusable — truncated JSON, or
    # the whitespace degeneration this endpoint shows a few percent of the
    # time (one request in a 16-wide probe emitted 117k characters of newlines
    # and burned the whole token budget). The retry re-rolls the seed, since
    # repeating the same one reproduces the same degenerate sample.
    call_retries: int = 2
    # Transport hiccups (a gateway returning a non-JSON body, a 5xx, a 429) are
    # counted separately: they produce nothing to inspect, so they must not
    # consume one of the `call_retries` re-rolls that exist to escape a bad
    # sample. Deterministic errors (401, 400) and timeouts are never retried
    # here -- the first will not start working, and the second would multiply
    # the time one stalled request can pin a worker.
    transport_retries: int = 3

    # solution shape
    min_steps: int = 3
    max_steps: int = 6
    # Show the benchmark's golden answer to the model. True = rationalisation
    # (the trace is guaranteed to land on the right answer, which is what makes
    # the data usable for SFT); False = honest solve, lower yield.
    reveal_answer: bool = True

    # execution of the generated plotting code
    max_repair_rounds: int = 2
    exec_timeout: float = 180.0
    dpi: int = 120
    figwidth: float = 6.0
    figheight: float = 6.0

    # answer checking
    num_rtol: float = 1e-2
    num_atol: float = 1e-2
    point_atol: float = 0.05
    # How close a *marked* point on the canvas has to be to the golden point
    # for the drawing to count as showing the answer. Looser than point_atol:
    # this is about the picture, not the arithmetic.
    draw_atol: float = 0.25

    # incrementality check: fraction of pixels a step must change w.r.t. the
    # previous frame for the step to count as "drew something new". Step 1 is
    # compared against the rendered preamble, not against a blank image — grid
    # and axes alone move ~3% of the pixels.
    min_pixel_delta: float = 1e-4
    # ...and how much of the previously-drawn ink a step may erase. Only
    # checked while the view box is unchanged, since expand_view legitimately
    # moves every pixel; the slack covers a legend that jumps to a new corner.
    max_ink_loss: float = 0.05
    # Reject solutions that paste the golden answer into the code as a literal
    # instead of deriving it. Only meaningful together with reveal_answer.
    reject_pasted_answer: bool = True

    # Stop the batch when the volume gets this close to full. The output lives
    # on a shared 98 TB NFS mount: on 2026-09-10 another tenant filled it
    # mid-run and 2471 of 2696 tasks turned into ENOSPC failure rows in six
    # minutes. Failing the run is fine; converting every remaining task into a
    # permanent "failed" record is not.
    min_free_gb: float = 5.0

    # Circuit breaker. An exception taxonomy keeps losing the race with reality:
    # the endpoint can be down (connection refused), warming up (503), throttling
    # (429) or hung (timeout), and each shape brands every task failed at its own
    # pace. Whatever the shape, N generation failures in a row with nothing
    # succeeding in between means the environment died, not the tasks.
    max_consecutive_failures: int = 15

    note: str = ""


EXPERIMENT_CONFIGS: list[GenConfig] = [
    GenConfig(
        name="qwen38-27b",
        model="Qwen/Qwen3.8-27B",
        note="default arm: newest small dense model on the endpoint",
    ),
    GenConfig(
        name="qwen35-27b",
        model="Qwen/Qwen3.5-27B",
        note="previous generation of the same size, for a quality delta",
    ),
    GenConfig(
        name="qwen3-next-80b",
        model="Qwen/Qwen3-Next-80B-A3B-Instruct",
        note="MoE 80B/3B-active: cheap-per-token upper bound",
    ),
    GenConfig(
        name="gpt-oss-120b",
        model="Openai/Gpt-oss-120b",
        note="largest model on the endpoint; teacher for distillation checks",
    ),
]

CONFIGS_BY_NAME = {c.name: c for c in EXPERIMENT_CONFIGS}
DEFAULT_CONFIG = EXPERIMENT_CONFIGS[0].name

# Sample used for the smoke run: circumcenter of a triangle — genuinely
# incremental (points -> triangle -> perpendicular bisectors -> intersection ->
# circumscribed circle) and has a `point` answer schema.
#
# NOTE: `qid` is only unique *within* a task file (e.g. 6000-6099 is used by
# both `circle_properties/inner_circle` and
# `elementary_calculations/center_coordinates`), so a sample is addressed as
# "<file-stem>:<qid>".
SMOKE_TASK = "outer_circle_center:6200"


# ===========================================================================
# Coordinate-blind evaluation
# ===========================================================================
#
# Plan: `experiments/noregeo-vlm-eval.md`. Measures whether a VLM can solve a
# NoReGeo task when the coordinates are NOT in the text and the only source of
# them is the picture — and, with the `swap` arm, whether it is reading the
# picture at all or answering from the text it was also given.
#
# The benchmark already ships the vision arm: every row carries
# `image_question` / `image_answer` / `im_path` / `im_dots_path`, and no
# `image_question` contains coordinates (audited over all 2700 rows).

EVAL_OUT_DIR = REPO_ROOT / "run" / "noregeo_eval"

# `im_path` (images_full/) PRINTS THE ANSWER IN THE FIGURE TITLE — e.g.
# "Task 9000: area = 17.50" — so it is unusable for evaluation and this module
# never names it. `im_dots_path` (images/) is the clean one on 26 of the 27
# stems: the given points, labelled, on a grid, with the task parameter (the
# mirror axis, the rotation angle) in the title.
EVAL_IMAGE_FIELD = "im_dots_path"

# ...and the 27th. `special_lines` asks "is the segment from B to D a median,
# altitude, or bisector?" and titles its dots figure with the answer ("Task
# 10001: altitude") on 100 rows out of 100. The figure arms would be reading
# the caption, not the geometry, so they skip the stem; the text arm is fine.
FIGURE_ANSWER_IN_TITLE: tuple[str, ...] = ("special_lines",)

# Stems dropped from every arm, with the reason logged at selection time:
#
#   outer_circle_radius — the question asks for the circumradius (a length) and
#       the golden answer is the circum*center* (`schema_type: "point"`, 100 of
#       100 rows, verified against an independent solver). A model that answers
#       the question as asked is scored wrong, whichever arm it is in. This is
#       the same mislabelling as `triangle_type`, which asks for the incircle
#       radius under a triangle-classification name — there the question and
#       the answer at least agree.
#   triangle_type — byte-identical to `inner_circle_radius`, all 100 rows
#       (same qids, same questions, same answers, same figures). Keeping both
#       gives incircle radius double weight in every average.
EXCLUDED_STEMS: dict[str, str] = {
    "outer_circle_radius": "question asks for the circumradius, golden answer "
    "is the circumcenter",
    "triangle_type": "byte-identical duplicate of inner_circle_radius",
}

# The arms. `swap` is the falsification test: the model gets the ORIGINAL text
# question (coordinates and all) together with a figure of a *scaled* version
# of the same configuration. A model that reads the picture answers for the
# scaled configuration; one that pattern-matches the text answers for the
# original. `swap-control` is the same thing at scale 1.0 — it separates "the
# model follows the text" from "the model cannot parse our renderer".
ARM_TEXT = "text"  # text question with coordinates, no image
ARM_FIGURE = "figure"  # image_question + dots figure: coordinate-blind
ARM_BOTH = "both"  # text question + dots figure: does vision add anything
ARM_SWAP = "swap"  # text question + figure of a scaled configuration
ARM_SWAP_CONTROL = "swap-control"  # same, scale 1.0
ARMS: tuple[str, ...] = (ARM_TEXT, ARM_FIGURE, ARM_BOTH, ARM_SWAP, ARM_SWAP_CONTROL)

# Scale factor for the `swap` arm. 0.5 rather than 2.0 so that the scaled
# configuration stays inside the benchmark's fixed [-10.5, 10.5] view box: the
# swapped figure differs from the control by the *configuration*, not by the
# axis range, which would be a giveaway that needs no geometry to spot. The 141
# configurations that leave that box anyway get a view fitted to the union of
# both scales (`figures.union_view_box`) — fitting it to the unscaled points
# alone put the shrunken configuration partly or wholly off-canvas.
SWAP_SCALE = 0.5

# How each stem's golden answer transforms when the whole configuration is
# scaled by `s` about the origin. This is what makes the falsification arm
# possible without writing a solver per task type:
#
#   "area"      -> answer * s**2
#   "length"    -> answer * s
#   "point"     -> answer coordinates * s
#   (absent)    -> invariant, and therefore useless as a discriminator
#
# Classification stems are all invariant (collinear stays collinear, parallel
# stays parallel), so they are absent here and the swap arm skips them — 1100
# of 2700 rows. `reflection` and `rotation_point` stay in because reflection
# across a coordinate axis and rotation about the origin both commute with
# scaling about the origin.
SWAP_RULE: dict[str, str] = {
    "triangle_area": "area",
    "parallelogram_area": "area",
    "triangle_perimeter": "length",
    "parallelogram_perimeter": "length",
    "segment_length": "length",
    "inner_circle_radius": "length",
    "midpoint": "point",
    "center_coordinates": "point",
    "intersection": "point",
    "bisector": "point",
    "inner_circle_center": "point",
    "outer_circle_center": "point",
    "reflection": "point",
    "rotation_point": "point",
}

# Two stems ship a vision question with an unrendered template, so the datum
# the question turns on is missing from BOTH the text and the picture. The
# repair lifts it out of the *text* question:
#
#   shape_symmetry: "...symmetric about the {axis}-axis?" -> X or Y. The axis
#       is a given, not a coordinate, so the repaired row stays in the headline.
#   symmetry:       "...symmetric about None?" (50 of 100 rows) -> the centre,
#       e.g. "point O=(-4, -3)". That DOES put coordinates back into the
#       prompt, and the centre is not drawn in the figure either, so these rows
#       are reported separately and excluded from the headline number.
QUESTION_REPAIRS: dict[str, dict[str, object]] = {
    "shape_symmetry": {
        "placeholder": "{axis}",
        "pattern": r"symmetric about ([XY])-axis",
        "injects_coordinates": False,
    },
    "symmetry": {
        "placeholder": "None",
        "pattern": r"symmetric about (point O=\([^)]*\)|line [^?]+)",
        "injects_coordinates": True,
    },
}

# Stems whose *figure* cannot carry the question even after the repair. The
# shape_symmetry dots image plots the polygon's vertices unlabelled and without
# edges, so "is the polygon symmetric" additionally requires guessing the
# vertex order. Kept in the run, reported apart from the headline.
FIGURE_UNDERSPECIFIED: tuple[str, ...] = ("shape_symmetry",)

# Far below `DEFAULT_WORKERS`. The endpoint's *priority* throttle (429 "Too
# many requests to model. Wait the owner") sheds eval traffic at a concurrency
# the generation run never hit: on 2026-09-15 four workers took 429s on every
# request for minutes at a time. Two is a request in flight plus one waiting,
# which is what a shared endpoint under someone else's priority allows.
DEFAULT_EVAL_WORKERS = 2


@dataclass(frozen=True)
class EvalConfig:
    """One evaluated model, plus how its answers are scored."""

    name: str
    model: str
    base_url: str = DEFAULT_BASE_URL

    # Greedy: this is a measurement, not a generation run, so we want the
    # model's single best answer rather than a sample from its distribution.
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 2048
    # Reached only by the retry path in `generate._call`, which escalates the
    # budget when a completion was cut off rather than degenerate.
    max_tokens_ceiling: int = 6144
    seed: int | None = 1234
    thinking: str | None = "low"
    request_timeout: float = 300.0
    call_retries: int = 2
    # Higher than the generation run's 3: the failures here are 429s from a
    # shared endpoint rather than gateway hiccups, and giving up on one turns a
    # throttled minute into a permanently unscored row.
    transport_retries: int = 6
    # ...and each wait is allowed to grow to a minute, because the throttle it
    # is waiting out lasts minutes, not seconds.
    retry_backoff_cap: float = 60.0
    max_consecutive_failures: int = 15
    min_free_gb: float = 5.0

    # Scoring. The strict tolerances are the ones `verify.py` already uses, so
    # a figure-arm answer is held to the same standard as a generated trace.
    num_rtol: float = 1e-2
    num_atol: float = 1e-2
    point_atol: float = 0.05
    # ...and a loose pass recorded alongside it. The difference between the two
    # separates "read the picture slightly wrong" from "did not read it": a
    # model that misplaces one vertex by a grid square lands here, not at zero.
    loose_num_rtol: float = 0.05
    loose_num_atol: float = 0.5
    loose_point_atol: float = 0.75

    note: str = ""


EVAL_CONFIGS: list[EvalConfig] = [
    EvalConfig(
        name="qwen38-27b",
        model="Qwen/Qwen3.8-27B",
        note="the model that generated the synthesis dataset",
    ),
    # `thinking=None` for the Medgemma arms: they are not reasoning models, and
    # `reasoning_effort` on a model that does not take it is a 400 — which the
    # retry policy (correctly) does not retry.
    EvalConfig(
        name="medgemma-27b",
        model="Google/Medgemma-27b-it",
        thinking=None,
        note="best coordinate read-off in the 2026-09-15 pilot",
    ),
    EvalConfig(
        name="medgemma-4b",
        model="Google/Medgemma-1.5-4b-it",
        thinking=None,
        note="small VLM: how much of this is model size",
    ),
    EvalConfig(
        name="qwen35-27b",
        model="Qwen/Qwen3.5-27B",
        note="previous generation; returned empty completions on image input "
        "in the pilot, so expect a high unparseable rate",
    ),
]

# `Openai/Gpt-oss-120b` is deliberately NOT in the grid: the endpoint accepts
# an image payload for it and the model then answers "I can't see images", so
# its figure-arm score would measure the gateway, not the model. It is useful
# as a blind control (`--config gptoss-120b-blind`), and only as that.
BLIND_CONTROL = EvalConfig(
    name="gptoss-120b-blind",
    model="Openai/Gpt-oss-120b",
    note="NOT multimodal despite accepting the payload: blind control only",
)

EVAL_CONFIGS_BY_NAME = {c.name: c for c in [*EVAL_CONFIGS, BLIND_CONTROL]}
DEFAULT_EVAL_CONFIG = EVAL_CONFIGS[0].name
