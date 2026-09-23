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
All 139 NoReGeo tests passed before submission. Evaluation is pending.

## Conclusions

Pending; report new-cohort outcomes separately before pooling.

## Changelog

- 2026-09-23 — Added disjoint outcome-blind cohort and combined HTML/token report.
- 2026-09-23 — Initial worker encountered TLS connection failures after first
  successful calls. AIRI served its leaf without the GlobalSign intermediate.
  The public intermediate was fetched from its CA's certificate URL and verified
  against the system root store (`openssl verify: OK`). A per-job CA bundle keeps
  certificate/hostname verification enabled. Explicit connection recovery archives
  the old attempt and preserves successful request caches and completed arms.
