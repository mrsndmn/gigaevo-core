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
* **`im_dots_path` leaks the answer too — on one stem.** `special_lines` asks
  "is the segment from B to D a median, altitude, or bisector?" and titles its
  dots figure with the answer ("Task 10001: altitude") on 100 rows out of 100.
  The figure arms skip that stem; the text arm keeps it.
* **`outer_circle_radius` asks a question its golden answer does not answer.**
  The question says "Find the circumradius" — a length — and all 100 rows carry
  `schema_type: "point"` with the circum*center* as the answer. Excluded from
  every arm. It also means the 91 accepted `outer_circle_radius` traces in the
  synthesis dataset derive a circumcenter under a question that asks for a
  radius.
* **`triangle_type` is a byte-identical duplicate of `inner_circle_radius`**,
  all 100 rows. Excluded, so incircle radius is not counted twice — and the
  synthesis dataset's 199 accepted traces across the pair cover 100 distinct
  problems, not 200.
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

Two traps inside that idea, both caught by review before any numbers were
taken:

* the shared view box has to hold **both** scales. Fitted to the unscaled
  configuration alone, scaling towards the origin put at least one point
  off-canvas on 57 of the swap-eligible tasks and produced an entirely empty
  grid on 11 of them — which the arm would then have scored as "the model did
  not read the figure", for a figure with nothing in it;
* a loose tolerance only discriminates while the two goldens are further apart
  than the tolerance. On incircle radii (1–3, halving to 0.5–1.5, against
  `loose_num_atol = 0.5`) they are not, and the row would count in the
  `figure~` and `text~` columns at once. Those rows are now counted under
  `ambig` instead.

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

### The grid, 2026-09-16

3 tasks per stem, both models, every arm — 303 jobs each, finished 05:11.
Exact / loose, over the rows that survived the exclusions:

| model | text | figure | both | swap | swap-control |
|---|---|---|---|---|---|
| Qwen3.8-27B | 97.3 / 98.7 (75) | **88.1** / 91.0 (67) | 97.2 / 97.2 (72) | 40.5 / 83.3 (42) | 97.6 / 100.0 (42) |
| Medgemma-27b | 73.0 / 73.0 (74) | **19.7** / 24.2 (66) | 93.5 / 93.5 (62) | 0.0 / 22.0 (41) | 14.6 / 31.7 (41) |

Falsification — which source the answer follows, strict / loose:

| model | follows figure | follows text | neither (strict) |
|---|---|---|---|
| Qwen3.8-27B | 40.5 / **83.3** | 2.4 / **4.8** | 57.1 |
| Medgemma-27b | 0.0 / 22.0 | 0.0 / 4.9 | 100.0 |

Errors: 0 for Qwen, 14 for Medgemma (all "no usable completion after 3
attempts" — schema-invalid output, 10 of them in the `both` arm, which is why
that arm's 93.5% is over 62 surviving rows and not 72).

## Conclusions

**The gap is real and model-dependent.** Moving the coordinates from the text
into the picture costs Qwen3.8-27B 9 points (97.3 → 88.1) and Medgemma-27b 53
(73.0 → 19.7). A single number for "VLMs cannot read figures" would have been
wrong in both directions.

**Qwen3.8-27B is genuinely reading the figure, not recalling the benchmark.**
In the falsification arm it answers for the drawn configuration 83% of the time
and for the stale text it was also given 5% of the time. Recall of a public
benchmark and text-shortcutting both look like the second number, and it is not
there.

**What it lacks is metric precision, not comprehension.** 18 of the 42 swap
rows read the configuration correctly and still missed the strict tolerance,
by 0.23–0.48 — a quarter to half a grid square. The strict-to-loose jump
(40.5 → 83.3) is that error and nothing else. So the capability to buy with
training is *reading the grid accurately*, not *attending to the image*; the
figure arm's per-stem losses agree, being spread thin (1/3 on `parallel_lines`,
2/3 on seven others) rather than concentrated in a category.

**The swap-control earns its place, but it is not a clean single-variable
control.** It differs from `both` in two ways at once: the figure is ours
rather than the benchmark's, *and* the prompt carries the "the text is stale,
the image is authoritative" instruction. Medgemma answers 93.5% on `both` and
14.6% on `swap-control`, and that 79-point drop cannot be attributed to one of
the two by the arms as they stand.

The renderer is nonetheless exonerated by a paired comparison against the
`figure` arm, on the tasks both arms scored:

| model | figure (benchmark's PNG) | swap-control (ours) | n |
|---|---|---|---|
| Qwen3.8-27B | 95.2% | 97.6% | 42 |
| Medgemma-27b | 7.5% | 15.0% | 40 |

Our figure is no harder than the benchmark's for either model — Medgemma
actually does better on it. So what the instruction does is push the model off
the text and onto a read-off it cannot perform, and Medgemma's own `reasoning`
shows exactly that: on `center_coordinates:6017` it states "the coordinates of
point A are (-4, 8) and of point B are (-2, -2)" and computes a midpoint of
(-3, 3) against a golden (-3.5, 3.5) — it is not refusing and not blind, it is
misreading. Its swap row therefore says nothing about a text-vs-figure
preference: it cannot read either figure accurately enough to express one.

Caveats: 3 tasks per stem is a coarse sample; Medgemma's `both` arm lost 10
rows to unusable completions, which could flatter it; and in `swap-control` the
figure agrees with the text by construction, so a model that quietly ignores
the instruction and answers from the text still scores — which is why
Qwen's 97.6% there sits above its 88.1% on the `figure` arm. Separating the
renderer from the instruction needs a sixth arm (our figure, no instruction),
which this grid does not have.

After the exclusions the arms cover 2500 (text), 2400 (figure, both) and 1400
(swap, swap-control) of the 2700 rows; every skip is logged with its reason at
selection time.

Next: this is the baseline for the coordinate-blind regeneration — traces where
the model sees the figure and the question carries no coordinates, filtered by
the same automatic checker as the text-arm dataset.

## Changelog

- **2026-09-15** — audit of the benchmark's vision fields (answer-leaking
  `im_path`, two unrendered question templates); `eval_blind.py`, `figures.py`,
  the eval grid in `configs.py` and `tests/noregeo_synth/test_eval_blind.py`;
  pilot on 2 tasks × 5 arms.
- **2026-09-16** — adversarial review, three defects fixed before the grid ran:
  `outer_circle_radius` crashed the swap arm on all 100 rows (its answer is a
  point, not a length) and now sits in `EXCLUDED_STEMS` along with the
  duplicate `triangle_type`; `special_lines` titles its dots figure with the
  answer, so the figure arms skip it; the swap view box is the union of both
  scales, which un-blanks 57 figures. Plus: `--stem` no longer silently
  evaluates every other stem, and the loose falsification columns skip the rows
  where the two goldens are closer than the tolerance.
- **2026-09-16** — the grid ran: 3 tasks per stem × 5 arms × 2 models, 606 jobs,
  finished 05:11. Qwen3.8-27B loses 9 points going coordinate-blind and follows
  the drawn configuration over the stale text 83% to 5%; Medgemma-27b loses 53
  and cannot use the rendered figure at all.
