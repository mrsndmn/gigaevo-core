# Full NoReGeo feedback comparison

## Hypothesis
Larger coverage reveals answer and construction failures, and tasks where retaining canvas observations changes success.

## Setup
Entry point: `experiments.noregeo_synth.recovery_full.run`; launcher: `python -m experiments.noregeo_synth.run_recovery_eval --experiment full`.
The experiment registry and inherited recovery protocol are the source of configuration.
All benchmark tasks are evaluated; known duplicate/mislabeled stems are reported separately from the primary score.
Successful responses are cached. Transport failures stop safely after bounded retries; unknown interrupted requests require inspection rather than silent regeneration.

## Results
Launched on MLS: `lm-mpi-job-29341cb1-3059-4746-9a15-743fe2c636b4`.
Staged commit: `d87f58cde547c0d78a1ea9b9998fe6f7a4a82881`.
Startup verified: running, real responses and isolated execution artifacts saved;
the first two completed measurements were correct. No aggregate conclusions yet.
Artifacts: `/mnt/virtual_ai0001053-00054_SR004-nfs2/d.tarasov/gigaevo-artifacts/results/noregeo_feedback/qwen38-recovery-v2-full2700`.
Live `analysis.html` lists failed tasks and links to embedded traces.
All 165 tests and independent review passed before submission.

## Conclusions
Pending measurements. Distinguish wrong answers, construction failures and infrastructure interruptions.

## Changelog
- 2026-09-23: Added full-dataset three-arm launcher, checkpoints and HTML failure browser.

- 2026-09-23: Submitted full run and verified live inference/execution startup.
