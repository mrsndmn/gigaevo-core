# NoReGeo: additional recovery cohort

## Hypothesis

The recovery pilot's equal solve rates and token-cost difference persist on
new tasks under the same protocol.

## Setup

`experiments.noregeo_synth.recovery_extra.run`, with the additional-cohort registry
in `recovery_extra.py:experiment_configs`, importing the unchanged recovery
protocol. Selection excludes all first-pilot IDs and balances type coverage,
prioritizing previously uncovered stems without consulting answers or outcomes.
Old records and their hashes are frozen for combined reporting.

This is a separately user-authorized finite cohort, not an override of the
original pipeline's stop gate. No format diagnostics, full-cohort continuation,
or injection experiment are launched. Both arms reuse a common first response.
Original launch/prepare/run/report arms remain available. The launcher selects
this experiment with `--experiment extra15`; `--dry` never submits.

HTML reports show the old pilot, new cohort and combined set, plus saved traces.
Usage accounting includes failed attempts and explicitly deducts duplicated
shared-first costs from overall spend. Full planned denominators remain visible.

## Results

Submitted MLS job `lm-mpi-job-aad6051c-07e6-4836-8f42-3f24b0984a1e`
from reviewed commit `eefb5c031a93fb73ef3d7cf4305867a791e71be8`.
All 139 NoReGeo tests passed before the initial submission; the subsequent TLS
recovery changes passed 142 tests and independent review.

The additional cohort finished with 15/15 correct in both arms. History used
208,375 input + 31,523 output = 239,898 tokens; canvas used 281,956 input +
28,705 output = 310,661 tokens (+29.5%). Action failures: 4 history, 6 canvas,
all premature finish requests subsequently corrected; no Python syntax failures.
All 30 measurements completed, with no residual request errors.

Combined disjoint cohorts: 30/30 correct in each arm. History used 491,512 tokens,
canvas 645,066 (+31.2%); action failures 9 versus 15. Output usage includes
reasoning; shared first calls are included in both arm totals for comparison.
Deduplicated physical request usage is 507,784 tokens for the new cohort and
1,046,087 across both cohorts, excluding the old format diagnostic. Independent
sum of all 143 new-cohort request-cache usage records equals the reported total.

Resume job `lm-mpi-job-1c1954e2-1f76-4589-b75d-b512987715fc` finished the worker
pipeline in 734.48 seconds, excluding the earlier TLS failure and investigation.
All six successful pre-recovery cached responses and the completed history arm
were verified byte-identical after completion. HTML copies are available at
`run/noregeo_feedback/extra15/comparison.html` and `analysis.html`.

## Conclusions

On these tasks, images add token cost without increasing final accuracy. Both
arms are at the observed accuracy ceiling, so this does not establish absence
of visual benefit on harder tasks or controlled construction corrections. The
new cohort added ten previously uncovered stems; both sets together cover all
25 available stems, with a small number of examples per stem.

## Changelog

- 2026-09-23 — Added disjoint outcome-blind cohort and combined HTML/token report.
- 2026-09-23 — Initial worker encountered TLS connection failures after first
  successful calls. AIRI served its leaf without the GlobalSign intermediate.
  The public intermediate was fetched from its CA's certificate URL and verified
  against the system root store (`openssl verify: OK`). A per-job CA bundle keeps
  certificate/hostname verification enabled. Explicit connection recovery archives
  the old attempt and preserves successful request caches and completed arms.

- 2026-09-23 — Archived connection failures and resumed as MLS job
  `lm-mpi-job-1c1954e2-1f76-4589-b75d-b512987715fc`, commit
  `1da3459079d9e86b2f1228d8f005676a229b3bb8`; successful replies reused.

- 2026-09-23 — Completed additional cohort, reconciled token accounting and
  preserved-cache checks, and published separate/combined HTML tables.
