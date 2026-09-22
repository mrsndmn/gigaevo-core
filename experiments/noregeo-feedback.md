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

Both arms branch from the same first action and retain the full action history.
Generated snippets execute locally through the existing guarded subprocess;
inference uses the configured remote endpoint. Prefix replay is checked against
previously observed frame hashes. Execution failures terminate the rollout.
Gold answers and final verification are offline only, never repair feedback.
Records retain unsuccessful rollouts as well as successful ones.

This tests adding rendered feedback when initial coordinates are already known.
It does not force visual dependence. A one-task demonstration establishes that
the protocol works; it cannot establish an improvement in solve rate. A larger
paired comparison must count execution failures as failures, report endpoint
errors separately, and report both answer correctness and full trace acceptance.

## Results

Pending local pilot.

## Conclusions

Pending pilot. No claim of a benefit from image feedback yet.

## Changelog

- 2026-09-23 — Added paired stepwise rollouts and local feedback-isolation tests.
