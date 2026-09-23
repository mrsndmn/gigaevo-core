# Full NoReGeo feedback comparison

## Hypothesis
Larger coverage reveals answer and construction failures, and tasks where retaining canvas observations changes success.

## Setup
Entry point: `experiments.noregeo_synth.recovery_full.run`; launcher: `python -m experiments.noregeo_synth.run_recovery_eval --experiment full`.
The experiment registry and inherited recovery protocol are the source of configuration.
All benchmark tasks are evaluated; known duplicate/mislabeled stems are reported separately from the primary score.
Successful responses are cached. Transport failures stop safely after bounded retries; unknown interrupted requests require inspection rather than silent regeneration.

## Results
Pending launch. Live HTML lists failed tasks and links to embedded traces.

## Conclusions
Pending measurements. Distinguish wrong answers, construction failures and infrastructure interruptions.

## Changelog
- 2026-09-23: Added full-dataset three-arm launcher, checkpoints and HTML failure browser.
