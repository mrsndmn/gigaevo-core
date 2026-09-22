# NoReGeo feedback: comparison with the historical evaluation

## Hypothesis

Rendered intermediate feedback improves final-answer correctness over the same
stepwise policy with code history and execution status. A fresh text-only
baseline distinguishes historical changes from the new rollout protocol.

## Setup

`experiments.noregeo_synth.feedback_eval.run`, with the entire design in
`feedback_eval.py:experiment_configs` and imported feedback/evaluation configs.
The cohort is every task in the frozen historical text arm, independent of its
outcome. Input questions and gold answers must match the cached benchmark;
the historical file checksum is pinned in code. No answer-revealing synthesis
acceptance rates are used as solve-rate baselines.

Primary comparison: paired strict final-answer correctness, including execution
failures as wrong. Report discordant pairs and an exact two-sided McNemar test.
Request/infrastructure failures remain explicit against the full planned cohort.
No stopping based on interim significance. Full trace acceptance is secondary.
Historical comparisons use matched task IDs per arm; changing protocols and
budgets prevents a causal interpretation. Swapped configurations are excluded.
The fresh text baseline repeats the previous text-evaluation protocol.

Preparation freezes tasks and historical rows outside the working tree. The MLS
launcher stages committed code, deduplicates active jobs and skips a completed
experiment. Resume keeps terminal records and the common first action, archives
unfinished attempts, and retries missing/error measurements.

Commands:
- `python -m experiments.noregeo_synth.run_feedback_eval --dry`
- `python -m experiments.noregeo_synth.run_feedback_eval`
- `python -m experiments.noregeo_synth.feedback_eval --report --out <artifact directory>`

The job writes `comparison.html` after each task and a separate HTML trace for
each task. Artifact paths and allocation selection are defined in code; the
existing project NFS artifact root is reused. Model calls go to AIRI; MLS hosts
the orchestrator and drawing subprocesses. The token is read from the existing
shared credential file, not embedded in manifests or staged code.

## Results

Submitted MLS job `lm-mpi-job-9b114238-dacb-4986-9d57-c6db23fc24c0`
from commit `bb6bafe0f8fd49caafdac2863d52683edcc8e574`. The job is running;
the first fresh text baselines were saved successfully and paired rollouts
have started. Final metrics are pending.

## Conclusions

Pending evaluation. The earlier one-task smoke test establishes operation only.

## Changelog

- 2026-09-23 — Planned matched historical comparison, fresh text baseline,
  resumable paired rollouts, HTML summaries and MLS launcher.
- 2026-09-23 — Review fixes: drawing timeouts are terminal failures, resume
  recovers the first action from both arms and rejects mismatches, concurrent
  report writers use unique temporary files.

- 2026-09-23 — Submitted and verified worker startup; first endpoint responses saved.
