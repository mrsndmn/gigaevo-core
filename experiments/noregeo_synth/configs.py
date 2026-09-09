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
