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

MLS job `lm-mpi-job-9b114238-dacb-4986-9d57-c6db23fc24c0`, staged commit
`bb6bafe0f8fd49caafdac2863d52683edcc8e574`, stopped with failed status after
attempting every task. No retries were launched during the error audit, per
user instruction. Fresh text reproduced the historical 73/75 correct.
History: 56 correct completed, 16 execution failures, 3 unresolved errors.
Canvas: 54 correct completed, 1 incorrect completed, 18 execution failures,
2 unresolved errors. Of the five unresolved errors, two are HTTP 500 and three
are malformed JSON after the configured attempts; they are not all transport
failures. No complete marker was written.

Offline audit of 34 execution failures: 25 Python syntax errors (history 12,
canvas 13), 3 failures of 2D np.cross (1/2), 2 unsupported mathtext commands
(1/1), 3 Python/Matplotlib API mistakes (1/2), and 1 inconsistent linear system
caught by a generated assertion (history). Of the syntax errors, 22 contain
short incomplete fragments at indexing/quotation boundaries and 3 have an
extra closing brace. In 23 of 25 cases total completion usage is below the
per-attempt budget. The client only JSON-decodes content; it does not truncate
code strings. Raw responses and finish reasons were not retained, so attribution
to model versus server structured decoding remains unresolved. Worker NumPy
source confirms that its cross implementation requires length-three vectors.

Among 70 terminal pairs, 43 are both correct, 12 history-only, 8 canvas-only,
and 7 neither; exact McNemar p=0.5034. Nineteen of the twenty discordant pairs
are execution failure versus a correct completed answer. Both arms complete
on 44 tasks: history 44 correct, canvas 43. This selected subset is diagnostic,
not an unbiased solve-rate comparison. The remaining disagreement is
semicircle_triangle:11066: canvas tests an obtuse-triangle semicircle-containment
interpretation instead of the right-angle criterion consistent with the gold.

The correct history answer for intersection:7273 fails only the secondary
visible-change check because one step only computes and prints. Primary
answer scoring correctly keeps it as a success.

Local standalone audit report (41 records, including secondary-check failures):
`run/noregeo_feedback/error_analysis/report.html`. It embeds saved frames and
source steps and performs no inference or generated-code execution.

## Conclusions

This run does not isolate geometric improvements from visual feedback well:
most discordances concern whether generated code survives to the final answer.
The protocol aborts on the first execution failure without returning the error
to the policy, so it does not test recovery from a failed action. Successful
completion on a selected subset cannot establish that images help or do not help.

Before another run, retain raw responses and finish reasons, diagnose code-string
serialization/decoding, and make the supported numerical/rendering API explicit.
Testing recovery would require a new protocol with equal repair budgets in both
arms. Preserve this run unchanged; no measurements were retried in this audit.

## Changelog

- 2026-09-23 — Planned matched historical comparison, fresh text baseline,
  resumable paired rollouts, HTML summaries and MLS launcher.
- 2026-09-23 — Review fixes: drawing timeouts are terminal failures, resume
  recovers the first action from both arms and rejects mismatches, concurrent
  report writers use unique temporary files.

- 2026-09-23 — Submitted and verified worker startup; first endpoint responses saved.

- 2026-09-23 — Audited saved errors without retries; separated malformed JSON
  from HTTP failures and documented execution confounds and offline HTML report.
- 2026-09-23 — Subsequent recovery-protocol review found an additional v1
  confound: shared canvas view bounds included the golden point via
  `dataset.coordinate_pairs`. Eight of the historical cohort's input views
  change when gold is removed. V1 artifacts remain unchanged; v2 builds the
  policy canvas from allowlisted input fields only. Interpret v1 visual results
  with this information leak in mind; the plain-text baseline is separate.
