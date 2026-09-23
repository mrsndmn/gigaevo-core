# NoReGeo: latest-frame context

## Hypothesis

Keeping only the latest canvas state reduces input-token cost without reducing
accuracy compared with keeping every historical image.

## Setup

`experiments.noregeo_synth.recovery_last.run`; all settings are in its
`experiment_configs` and the unchanged imported recovery protocol. The cohort
is exactly the union of the two completed recovery cohorts. History and canvas
records are frozen, not rerun; the new arm reuses their identical first response.

`recovery_eval.context_for` filters image attachments at every request boundary,
including repair and final-answer requests. All text messages are retained.
After failed execution the latest committed-state image remains. Successful
replacement updates the image. Canonical traces and saved frames are not erased.
A report audit rejects requests containing multiple images in the new arm.

Launcher: `python -m experiments.noregeo_synth.run_recovery_eval --experiment last30`;
`--dry` submits nothing. Worker has prepare/run/report modes. Existing original
and extra-cohort launch/worker/report behavior remains unchanged.

Reports compare all three arms on the fixed cohort, preserving failures in the
planned denominator. Per-arm costs include the reused first response for fair
comparison; additional physical usage excludes it. Raw API replies, usage and
request payloads remain available. This is a sequential comparison with previous
runs, not a concurrent control for endpoint model/version drift.

## Results

Pending.

## Conclusions

Pending; equal accuracy on an easy cohort does not demonstrate visual benefit.

## Changelog

- 2026-09-23 — Added latest-frame arm, frozen three-arm comparison and HTML audit.
