---
status: accepted
date: 2026-08-28
---

# 3. Claims are made at transplant scale; the 33M setup is a rung, not a venue

## Context

The programme's centre of mass moved between studies. Studies 1 and 3 are about
the **tokenizer** — segmentation, surface factorization, byte-exact
invertibility — measured on 33M-parameter models trained from scratch. Studies
2 and 4 are about the **embedding/output parameterization** — what function maps
ids to vectors and back — measured by transplant onto a frozen Qwen3-1.7B.

Study 2 finding 5 records why the 33M venue is receding: at 33M every baseline
had d upper ≈ 2.0, while pretrained Qwen3-1.7B has d upper = 0.305. "A large
pretrained model has already learned most of the case robustness the
factorization supplies by construction." Study 1 finding 1 independently
records that at 33M "vocabulary size barely moves BPB".

One tokenizer thread was still open: the joint (stem × case) softmax, deferred
from study 3 and described there as "the only remaining candidate that could
close" the +1.5% clean-BPB gap between `fact8k-v2` (1.2056) and `bpe8k`
(1.1881). It is cheap — ~13 min on the 33M harness.

## Decision

The programme's core entity is the **embedding/output parameterization**, and
claims are made at **transplant scale** on the frozen 1.7B body. The 33M
from-scratch setup is retained only as **rung L2 of the fitness ladder** — an
instrument, not a venue for conclusions.

`stem×case-softmax` is therefore **deferred**, not killed. Studies 1 and 3 close
as written, with their +1.5% asterisk intact.

## Consequences

- A ~13-minute experiment that would have closed a stated open thread is
  deliberately not run. This is the cost of the decision and it is accepted:
  the quantity it would resolve is a 1.5% BPB gap at a scale that no longer
  carries claims.
- Studies 1 and 3 keep a permanent open question in the record. Their headline
  — 4x smaller vocabulary at parity BPB with 2.5x less case damage — stands,
  but "reaches parity with plain BPE" remains unproven at 33M.
- The 33M harness must still be maintained and revived, because
  `ladder-calibration` depends on L2. Deferring the claim does not retire the
  code.
- The case-factorization idea is not lost: case shape is already a routing
  feature in `token_features.py`'s feature set v0, so it re-enters the
  programme at transplant scale by a different route.
