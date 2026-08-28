---
status: accepted
date: 2026-08-28
---

# 1. Rank vocabulary-programme experiments by information per GPU-hour, not by framework value

## Context

The vocabulary-factorization programme lives in `experiments/` inside GigaEvo,
an LLM-driven evolutionary program-search framework. Its stage-1 plan ends by
naming `token_features.py` "the genome" and specifying an L0–L3 fitness ladder
for a GigaEvo evolution phase.

That creates two defensible readings of what the programme is *for*, and they
order the open tracks in almost opposite ways:

- a **research programme** about vocabulary and embedding parameterization,
  which happens to be hosted here; or
- a **capability demonstration** for GigaEvo, in which every measurement track
  is scaffolding for the evolution phase.

The evidence pointed both ways. The framework reading is supported by the repo
and by the plan's own genome/ladder language. The research reading is supported
by how the four completed studies were actually run: pre-registered kill
criteria, load-bearing controls (`orig-reinit`), a measured run-to-run noise
floor (0.0016 BPB), and two of four headline results kept as negatives.

## Decision

The ranking objective is **information gained per GPU-hour toward a claim that
survives scrutiny**. A cheap probe that can return a decisive negative outranks
an expensive one that can only return a positive. Negative results count as
success. The GigaEvo evolution phase is ranked on the same scale as everything
else and earns its budget rather than receiving it by default.

## Consequences

- Tracks that can *falsify something already written down* rank above tracks
  that can only add a new claim. This is what puts `transplant-hardening`
  first, ahead of the flagship `stage1-tree`.
- Expensive tracks must be gated by cheap ones. `evolution-phase` is blocked
  behind `ladder-calibration` (see ADR 0002) rather than started on faith.
- The programme will not optimize for producing a GigaEvo demo. If the
  evolution phase never clears its gate, that is an acceptable outcome of this
  programme and not a failure of it.
- Compute abundance does not change the ordering. The allocation makes tracks
  *affordable*; it does not make an unvalidated 173 GPU-h expenditure wise.
