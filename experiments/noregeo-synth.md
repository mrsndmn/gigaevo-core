# NoReGeo → synthetic "reasoning + incremental figure" traces

## Hypothesis

NoReGeo (`FusionBrainLab/NoReGeo`) ships 2700 geometry problems with golden
answers but no worked solutions: the reasoning that connects the question to
the answer is exactly what is missing. If a mid-sized instruct model is made to
emit its solution as *executable construction steps* — one prose step paired
with one matplotlib fragment that adds that step's objects to a shared canvas —
then the benchmark's golden answer becomes an automatic filter, and what comes
out is a supervised trace where every reasoning step has a picture of the state
of the construction at that step.

The claim being tested is that the filter is strong enough to be worth it: that
"the code ran, every step changed the picture, the code's own answer equals the
golden answer, and the answer is drawn where it belongs" rejects the traces
whose prose is right by accident.

That last clause is not decoration. In the very first generated sample the
prose derived the circumcenter (45/13, −7) correctly and the code drew it at
(−2.46, 4.0) — a sign flip in the perpendicular direction. Answer-only checking
accepts that sample and poisons the dataset with a figure that contradicts its
own caption.

## Setup

`experiments/noregeo_synth/`, driven by
`python -m experiments.noregeo_synth.generate` (see `configs.py` — the grid,
the models, sampling knobs and all tolerances live there, and nothing about
them is repeated here).

Shape of the pipeline, per sample:

1. **Canvas** (`canvas.py`) — *we* generate the preamble: one figure, one axes,
   view box from the task's coordinates, the task's named points bound as `P`,
   plus `expand_view` / `mark` helpers. The exact source is pasted into the
   prompt, so the model codes against a known environment instead of inventing
   one.
2. **Generation** (`prompts.py`) — a JSON-schema-constrained completion: a plan,
   3–6 `{title, reasoning, code}` steps, and a typed `final_answer` whose schema
   is built from the row (enum for classification, `{x, y}` for point, number
   otherwise).
3. **Execution** (`execute.py`, `_runner.py`) — the fragments run in order in
   one `python -I` subprocess, so the namespace and the figure persist and step
   *k*'s PNG contains steps 1..*k*. A static guard (`guard.py`) rejects, by
   name, imports outside numpy/math/matplotlib, filesystem calls, and the
   constructs that break the shared canvas (`ax.clear()`, `del`, `set_xlim`,
   `set_size_inches`, `getattr`, ...). It is a name-level check, not a sandbox:
   what actually contains the code is the separate process and the timeout, and
   what actually enforces the canvas contract is the pixel checking below.
4. **Acceptance** (`verify.py`) — independent signals, all of which must pass:
   * the code ran;
   * every step changed the picture *inside the axes rectangle* — step 1 is
     diffed against the rendered preamble (a blank baseline would wave through
     a step that draws nothing, since grid and axes alone move ~3% of the
     pixels) and the title band is excluded (it changes every step by
     construction);
   * no step erased earlier content (`ink_loss`), checked only while the view
     box is unchanged, since `expand_view` legitimately moves every pixel;
   * `final_answer` matches the golden answer;
   * so does `state["answer"]`, the answer the *code* computed;
   * for point tasks, a marked point sits at the golden coordinates;
   * the golden answer is not pasted into the code as a literal
     (`literal_leak`), ignoring numbers the problem statement already gave.

   Failures are fed back verbatim for up to `max_repair_rounds` retries.

### What the filter does and does not prove

With `reveal_answer=True` (the default) the golden answer is in the prompt, so
the filter cannot prove the model *could* solve the problem. What it proves is
that the trace is self-consistent: the construction in the code produces the
answer, the picture shows it, no step is decorative, and the number was derived
rather than retyped. That is the property a training trace needs; measuring
solve rate is a separate run with `--no-reveal-answer`.

Artifacts per sample: `step_XX.png`, `contact_sheet.png`, the benchmark's own
`reference_full.png`, `preamble.py`, `prompt.json`, `record.json`; the dataset
rows are appended to `records.jsonl`.

## Results

Smoke run, 2026-09-09, `Qwen/Qwen3.8-27B` on the internal AIRI endpoint, four
samples covering all three answer schemas (and one row where `points` is null):
4/4 accepted, 0 repair rounds, 932–3731 completion tokens per sample, 26–92 s
each. `outer_circle_center:6200` (circumcenter): 5 steps, drawn-answer distance
0.0, per-step pixel deltas 0.017 / 0.007 / 0.008 / 0.123 / 0.006 and ink loss
0.0 everywhere it is defined.

Two findings about the endpoint that shape the config:

* full hidden thinking is unusable here — the model spent all 12288 completion
  tokens on thought and returned no content at all;
* with thinking *disabled* the grammar-constrained sampler degenerates into
  whitespace right after the first JSON field (107k characters of newlines,
  unparseable). `reasoning_effort="low"` is the setting that produces clean
  JSON, and it is the default in `configs.py` for that reason.

## Conclusions

The loop works end to end and the drawing-level check earns its place on
sample #1. Next: scale to a few hundred samples across all 27 task files to get
per-category acceptance rates, and compare the arms in `EXPERIMENT_CONFIGS`
(27B vs 80B-A3B vs 120B) on acceptance rate per token — the interesting
question is whether the cheap model plus the filter beats the expensive model
without it.

## Changelog

- **2026-09-09** — pipeline written; smoke run on `outer_circle_center:6200`
  accepted. Added the code-answer and drawn-answer checks after the first
  sample produced correct prose with a mis-drawn circumcenter.
- **2026-09-09** — hardening pass after an adversarial review, plus
  `tests/noregeo_synth/`. The checks that were shown to be bypassable: step 1
  was diffed against a blank canvas (so `pass` "changed" 3% of the pixels and
  was accepted); a step resizing the figure produced `NaN`, which passed the
  dead-step test and wrote invalid JSON; erasing content counted as changing
  it. Now every frame is diffed against the one before it inside the axes
  rectangle (title excluded, legend excluded), ink loss is tracked, and the
  guard also blocks `del`, `getattr`, `set_xlim`, `set_size_inches`,
  `set_visible`. `records.jsonl` upserts by task instead of appending — the
  sample directory is rewritten on every run, so an append log left old rows
  pointing at a newer run's images. A `literal_leak` check rejects solutions
  that retype the golden answer instead of deriving it, one bad completion no
  longer aborts the batch, unchecked signals record as `None` rather than
  `true`, and `OPENAI_API_KEY` is only accepted when `OPENAI_BASE_URL` points
  at this endpoint (the repo's `.env` holds an OpenRouter key).
  Two false rejects surfaced and were fixed the same way they were found: the
  per-step title and a `loc="best"` legend jumping corners both counted as
  drawing/erasing until the comparison learned to exclude them.
