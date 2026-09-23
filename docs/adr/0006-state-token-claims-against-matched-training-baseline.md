---
status: proposed
date: 2026-09-24
---

# 6. Any claim that state tokens help is made against a matched-training baseline with a textual state table

Origin: unattended deep interview (`--auto` after Round 1); the decision is the
interviewer's recommendation, not a confirmed requirement.

## Context

The spec's go criterion is "the prototype after SFT beats B1 of the same model
on constructions of ≥ 5 moves", where B1 is the GeoBuildBench protocol
(LLM + DSL + kernel + textual feedback) with constrained decoding added. That
comparison confounds three things: training on construction trajectories at
all, having the current state available in the context, and the *form* of that
state. The spec's own risk 2 notes that a pointer head cannot reference a
missing object and so wins trivially; the same trivial win is available to
state tokens against a baseline that is not trained and never sees a state
summary.

## Decision

The comparator for V1 is a **matched-training baseline**: the same base model
(Qwen3-1.7B), the same generator trajectories, the same LoRA configuration and
the same constrained decoding over existing handles — with the geometric state
supplied after every move as a **textual table of objects** instead of state
tokens. The two arms differ in exactly one variable. Kill and success gates are
fixed before training: on held-out ≥ 5-move problems (n ≥ 300), go if the
success-rate gain is ≥ 10 points with a bootstrap 95% CI above zero; kill if it
is ≤ 3 points; anything between is a reconsider, not a pass.

## Consequences

- Training cost doubles relative to the spec's plan (two trained arms, each
  with two seeds), and the untrained GeoBuildBench-style protocol becomes a
  reference row, not the comparator.
- If the matched baseline closes the gap, GeoState's contribution narrows to
  token economy (H3) and out-of-distribution poses (H4), and the plan says so
  in advance rather than after the fact.
- Wrong-object-selection rate, not impossible-reference rate, is the H1
  metric, because constrained decoding removes impossible references in both
  arms.
