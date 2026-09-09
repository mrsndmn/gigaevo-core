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
   *k*'s PNG contains steps 1..*k*. A static guard (`guard.py`) rejects imports,
   file access, and anything that would break the shared canvas (`ax.clear()`,
   `plt.subplots()`, `.remove()`) before execution.
4. **Acceptance** (`verify.py`) — four independent signals: execution, a pixel
   delta per step (a step that changes nothing is a broken step), the model's
   `final_answer` against the golden answer, and the answer the *code* computed
   (`state["answer"]`) against the same golden answer; for point tasks the
   finished canvas is additionally scanned for a marked point at the golden
   coordinates. Failures are fed back for up to `max_repair_rounds` retries.

Artifacts per sample: `step_XX.png`, `contact_sheet.png`, the benchmark's own
`reference_full.png`, `preamble.py`, `prompt.json`, `record.json`; the dataset
rows are appended to `records.jsonl`.

## Results

Smoke run, 2026-09-09, `Qwen/Qwen3.8-27B` on the internal AIRI endpoint,
`outer_circle_center:6200` (circumcenter, point schema): accepted, 4 steps,
0 repair rounds, 3410 completion tokens, 86 s wall time, drawn-answer distance
0.0, per-step pixel deltas 0.041 / 0.012 / 0.014 / 0.020.

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
