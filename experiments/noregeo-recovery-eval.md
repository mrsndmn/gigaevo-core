# NoReGeo: action format and recoverable visual feedback

## Hypothesis

A reliable action channel and explicit recovery opportunities allow a paired
test of whether a rendered canvas improves solution or correction of a model's
own intermediate construction. Earlier execution failures confounded that test.

## Setup

`experiments.noregeo_synth.recovery_eval.run`; the experiment registry and
all parameters and technical advance gates are in `recovery_eval.py`.
Submission entry point: `python -m experiments.noregeo_synth.run_recovery_eval`.
`--dry` prints the manifest without writes or submission.

The pipeline freezes prior syntax-failure contexts and compares structured JSON
with separately fenced Python. It saves raw responses, finish reasons, usage,
and transport errors. This outcome-selected diagnostic is not a benchmark score.
Format selection and subsequent advance gates are fixed in code and depend only
on technical outcomes, never answer correctness.

The outcome-blind pilot samples across task stems. Both arms share their first
response; only the image attachment differs. Both receive stdout, computed state,
and errors. Failed actions leave no state changes: each candidate is replayed
in a fresh isolated process from accepted steps. The model can replace the last
committed action or finish early. Old evaluation outputs remain untouched.
The full historical cohort includes the pilot exactly once. There is no new
text-baseline run: the previous fresh-text results remain descriptive context,
not a matched-budget causal control for this new protocol.

A separate small midpoint intervention inserts identical wrong draft coordinates
in both arms, exposes the inserted code, and allows replacement. This measures
correction of a controlled construction error, not natural-task accuracy; the
sample is exploratory. The draft is generated from inputs, not gold answers.

HTML reporting reads artifacts only and embeds images. Final-answer correctness
retains failed attempts in the planned denominator. Report completion, repairs,
format errors, token usage and runtime separately. Infrastructure/request errors
remain explicit. No automatic rerolls of terminal measurements or interrupted
rollouts; cached successful requests are reused. Technical gate stops are saved
as `stopped.json`, distinct from `complete.json` (all planned attempts finished).
Artifacts use a new run under the existing external NFS results root.

## Results

Implementation and validation in progress; no new scores yet.

## Conclusions

Pending. Do not infer visual usefulness from completion-filtered subsets.

## Changelog

- 2026-09-23 — Added staged format diagnostic, transactional recovery pilot,
  technically gated full cohort, separate midpoint intervention and offline HTML.
- 2026-09-23 — Review fixes: policy task fields are allowlisted so gold cannot
  alter canvas bounds; a started request without a result blocks silent resume;
  SDK retries are disabled; finishing requires visible construction and a
  non-null computed answer, independently of whether that answer is correct.
  Historical diagnostic prompts intentionally remain exact saved contexts and
  may retain the old answer-derived view bounds; they measure formatting only.
