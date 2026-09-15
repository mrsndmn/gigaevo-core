# Coordinate-blind NoReGeo: can a VLM get the configuration out of the picture?

## Hypothesis

The synthesis dataset (`experiments/noregeo-synth.md`) cannot, on its own,
teach a model to use figures: the generator never saw an image. The question
text carries the coordinates, the preamble binds them as `P`, and the view box
is computed from them — so the PNGs are a *product* of the trace, not an input
to it. Train a VLM on that and the cheapest policy available to it is to ignore
the pixels.

Before building data where vision is load-bearing, measure the gap on the
benchmark as it stands. NoReGeo already ships the vision arm — every row has
`image_question`, `image_answer`, `im_path`, `im_dots_path` — and our full run
used only the text arm.

Two claims are tested:

1. **The gap is real.** Solve rate falls when the coordinates move from the
   text into the picture, and the difference is the capability we want to buy.
2. **The measurement is not fooled by memorisation or by shortcutting.** A
   model that scores well on the figure arm might be reading the picture, or it
   might be recognising a public benchmark it has seen in pretraining. The
   `swap` arm separates the two: the text keeps the original coordinates, the
   figure shows the same configuration scaled by `SWAP_SCALE`, the prompt says
   the figure is authoritative, and the answer is scored against *both*
   configurations. Following the figure requires reading it; following the text
   is what recall and shortcutting both look like.

## Setup

`experiments/noregeo_synth/eval_blind.py`, driven by

```bash
python -m experiments.noregeo_synth.eval_blind --config qwen38-27b --per-stem 5
python -m experiments.noregeo_synth.eval_blind --summary --by-stem
```

Arms, models, tolerances, the per-stem scale-equivariance rules and the
question repairs all live in `configs.py`; nothing about them is repeated here.
`figures.py` renders the perturbed configurations, `verify.answer_matches` does
the scoring — the same function, and the same tolerances, that decide whether a
generated trace is accepted.

### What the audit of the benchmark's vision arm turned up

Checked over all 2700 rows before a single request was sent:

* **`im_path` prints the answer into the figure title** ("Task 9000: area =
  17.50"). It is unusable for evaluation; `EVAL_IMAGE_FIELD` pins the eval to
  `im_dots_path`, and a test asserts `images_full/` is never fetched.
* `image_question` never contains coordinates — the vision arm is clean.
* `image_answer` equals `answer` on every row, so one scoring path serves both.
* Two stems ship an unrendered template: `shape_symmetry` asks about the
  "{axis}-axis" (100 rows) and half of `symmetry` asks about "None" (50 rows).
  The datum is in neither the text-free question nor the picture, so it is
  lifted from the text question. For `symmetry` that puts coordinates back into
  the prompt, so those rows are flagged and reported outside the headline.
* The `shape_symmetry` figure plots the polygon's vertices unlabelled and
  without edges, so the question additionally requires guessing the vertex
  order. Flagged as `figure_underspecified`, reported apart.
* The reflection axis and the rotation angle — parameters the `image_question`
  omits — turn out to be rendered into the figure *title*, so those stems need
  no repair.

### Why the falsification arm scales rather than perturbs freely

Moving a point to an arbitrary new position would need a solver per task type
to recompute the golden answer. Scaling about the origin does not: areas go as
`s²`, lengths as `s`, answer points as `s`, and every classification answer is
invariant — which is also why the 1100 classification rows are excluded from
the arm and logged as skipped. `SWAP_SCALE = 0.5` keeps the perturbed
configuration inside the benchmark's fixed view box, so the swapped figure and
its control share one set of axes and differ in the configuration alone; a
fitted view would have made them pixel-identical apart from the tick labels,
turning a geometry test into a tick-reading test.

## Results

Pilot, 2026-09-15, `Qwen/Qwen3.8-27B`, 2 tasks × 5 arms: the loop runs, the
repairs fire, the swap arm scores against both configurations.

The endpoint then went into a *priority* throttle — 429 "Too many requests to
model. Wait the owner" on every request, for minutes at a time — which shapes
three knobs: `DEFAULT_EVAL_WORKERS` is 2 (four workers had every request shed),
`transport_retries` is 6 rather than the generation run's 3, and
`retry_backoff_cap` is 60 s, because backing off by 8 s six times only burns
the retry budget inside a single throttled window. The run rides it out:
`eval_blind` aborts with exit 3 rather than branding tasks failed, and
`run/noregeo_eval/run_grid.sh` retries every 10 minutes.

Separate 10-task probe of raw read-off ability, same day: asked only to read
the labelled points off the figure, `Medgemma-27b-it` placed 48% of points
within ±0.5, `Qwen3.8-27B` 22% (and failed to return parseable JSON on 7 of 10
attempts). That is the capability everything else rests on.

Full grid: pending.

## Conclusions

Pending the full run. The two numbers that matter are the `text` → `figure`
drop and the `swap` arm's figure-vs-text split; the second is the one that says
whether a figure-arm score means reading or recall.

## Changelog

- **2026-09-15** — audit of the benchmark's vision fields (answer-leaking
  `im_path`, two unrendered question templates); `eval_blind.py`, `figures.py`,
  the eval grid in `configs.py` and `tests/noregeo_synth/test_eval_blind.py`;
  pilot on 2 tasks × 5 arms.
