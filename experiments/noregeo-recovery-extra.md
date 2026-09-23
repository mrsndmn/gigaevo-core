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

Pending launch and evaluation.

## Conclusions

Pending; report new-cohort outcomes separately before pooling.

## Changelog

- 2026-09-23 — Added disjoint outcome-blind cohort and combined HTML/token report.
