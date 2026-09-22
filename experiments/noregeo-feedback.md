# NoReGeo: feedback from intermediate constructions

## Hypothesis

Seeing the executed intermediate canvas helps a model choose subsequent
construction steps and answer correctly, beyond retaining its own code history
and knowing that execution succeeded.

## Setup

`experiments.noregeo_synth.feedback.rollout`; the complete experiment grid,
model, budgets and observation modes live in `feedback.py:experiment_configs`.
Run with `python -m experiments.noregeo_synth.feedback --task <stem>:<qid>`.
Read saved results with `--summary --out <run directory>`.
For a standalone visual comparison, run
`python -m experiments.noregeo_synth.feedback_report --run <run directory> --out <report.html>`.
The report embeds the figures, compares steps side by side, and separates human
notes (`--note`) from automatic checks. It requires no network access to open.

Both arms branch from the same first action and retain the full action history.
Generated snippets execute locally through the existing guarded subprocess;
inference uses the configured remote endpoint. Each rollout retains one live
Python process across policy calls, preserving all intermediate variables.
Execution failures terminate that rollout; recoverable request failures do not
prevent the remaining arms and tasks from running.
Gold answers and final verification are offline only, never repair feedback.
Records retain unsuccessful rollouts as well as successful ones.

This tests adding rendered feedback when initial coordinates are already known.
It does not force visual dependence. A one-task demonstration establishes that
the protocol works; it cannot establish an improvement in solve rate. A larger
paired comparison must count execution failures as failures, report endpoint
errors separately, and report both answer correctness and full trace acceptance.

## Results

The initial pilot on `outer_circle_center:6200` completed both arms with the
correct final answer, computed answer and marked center. The canvas arm had an
incorrect radius in one prose step despite drawing it correctly; the verifier
does not validate every prose statement. This initial pilot used prefix replay;
review subsequently required persistent execution. Re-executing the captured
responses through the persistent worker produced byte-identical model requests
(including all image payloads) and identical verification results. No additional
inference was used for this replay; both live and replay artifacts are under
`run/noregeo_feedback/pilot-20260923*`.

## Conclusions

The protocol produces inspectable stepwise traces. One successful paired task
does not establish a benefit from image feedback.

## Changelog

- 2026-09-23 — Added paired stepwise rollouts and local feedback-isolation tests.

- 2026-09-23 — Review: preserve a live drawing process; continue other measurements after recoverable rollout failures.

- 2026-09-23 — Added standalone HTML analysis of saved rollouts.
