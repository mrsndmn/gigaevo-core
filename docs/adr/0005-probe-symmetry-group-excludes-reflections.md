---
status: proposed
date: 2026-09-24
---

# 5. The probe's symmetry group is the orientation-preserving similarity group; reflections are an out-of-distribution evaluation only

Origin: unattended deep interview (`--auto` after Round 1); the decision is the
interviewer's recommendation, not a confirmed requirement.

## Context

The GeoState spec names Sim(2) — translations, rotations, reflections, uniform
scale — as the primary group, and notes (risk 4) that the two points of an
intersection have no reflection-equivariant order: ordering them by
orientation is equivariant under rotations and scalings but swaps under a
reflection. The spec offers two remedies: an unordered pair with symmetric
handles, or dropping reflections. The choice is baked into the kernel, the
generator, the encoder, the consistency metric and every trajectory.

## Decision

For the decisive probe the group is the **orientation-preserving similarity
group** (translation, rotation, uniform scale). The `meet` of two circles, or
of a circle and a line, yields an **intersection pair ordered by orientation**
relative to the meeting objects, so the order is preserved by every group
element and a move can name "the first intersection" unambiguously.
Reflection is measured as an out-of-distribution condition in evaluation and is
not part of the training symmetry.

## Consequences

- Equivariance claims in the probe are made for the smaller group; the paper's
  Sim(2) framing needs either the unordered-pair design or an explicit
  reflection-canonicalisation step, to be decided after the probe.
- If E0 shows reflection to be the *only* transform under which existing
  models break, the group is widened and the pair becomes unordered — the
  reverse cost is a kernel and generator change plus regenerated trajectories.
- Tasks whose statement depends on orientation ("clockwise", "to the left
  of") are naturally in scope, since orientation is invariant under this group.
