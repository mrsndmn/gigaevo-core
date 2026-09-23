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

Submitted MLS job `lm-mpi-job-630c93f6-bde6-4495-879a-8d665a2996ac`
from reviewed commit `a15638cbc62bed4ab19797374e966b34bf644afa`.
All149 NoReGeo tests passed. The worker completed all30 new-arm measurements
in 645.29 seconds. All30 final answers are correct, matching both frozen baselines.

History: 423,774 input + 67,738 output = 491,512 tokens; 9 failed actions.
Canvas: 580,448 input + 64,618 output = 645,066 tokens; 15 failed actions.
Canvas_last: 492,235 input + 66,074 output = 558,309 tokens; 11 failed actions.
All failed actions were recovered. New-arm failures: 6 premature finish requests,
2 malformed block envelopes, and 3 Python syntax errors.

Latest-frame total usage is 13.4% below full-canvas usage and 13.6% above history.
Per-arm totals include 90,491 reused first-response tokens for comparability;
new physical API usage is 467,818 tokens across127 requests. Independently
reconciled physical usage plus shared-first usage equals the reported total.
All127 request payloads were verified to contain exactly the PNG of the most
recent committed successful action, including repairs and final requests.
All60 baseline record hashes and the source snapshot remain unchanged.
Standalone report: `run/noregeo_feedback/last30/analysis.html`.

## Conclusions

Retaining only the latest image saved tokens without loss of final-answer
accuracy on this cohort. Trajectories also differed, so the full cost difference
includes their changed repair behavior. All three arms are at the observed
accuracy ceiling; this result does not establish visual benefit on harder tasks.

## Changelog

- 2026-09-23 — Added latest-frame arm, frozen three-arm comparison and HTML audit.

- 2026-09-23 — Completed30 latest-frame rollouts; reconciled usage and verified
  all request images against exact last committed frames and baseline hashes.
