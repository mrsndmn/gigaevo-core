---
status: accepted
date: 2026-08-28
---

# 2. Calibrate the fitness ladder before running the evolution phase

## Context

The stage-1 plan specifies a four-rung fitness ladder for evolving
`token_features.py`: **L0** (seconds) frequency-weighted explained variance of
mean-of-pieces vectors under the candidate partition; **L1** (~1 min) step-0
clean BPB under the active-vocab mask; **L2** (~13 min) one 33M from-scratch
run; **L3** (~1 h) a full transplant. It adds the guard that "no evolutionary
conclusion is accepted from L0/L1 alone".

Two facts make the ladder unsafe as written.

**L0 scores the quantity study 4 stage 0 disproved.** Stage-0 finding 4:
"BPB damage is not monotone in MSE across methods; what the body cares about is
not Frobenius distance." L0 is a Frobenius-family fit score, on
mean-of-pieces targets that stage 1 itself says must not be trusted for values.

**The trustworthy rungs cannot carry a real search.** A completed GigaEvo run
in this repo (`problems/tabular_regression/elite_intra_extra_20260524.py`)
records ~800 unique evaluations in ~20h45m. At L2 that is ~173 GPU-h — roughly
25x the programme's total spend across all four studies to date — and at L3 it
is ~800 GPU-h. So selection pressure must come from rungs the plan distrusts,
or from an expenditure nothing has justified.

The programme has already been misled by unvalidated cheap proxies twice, both
recorded in study 3's methodological correction: tokens/byte does not forecast
BPB, and per-arm `inflation` rewards a bad clean baseline.

## Decision

`ladder-calibration` becomes a **track in its own right**, ranked ahead of the
evolution run and gating it.

Score ~15–20 candidate feature sets at L0, L1 and L2, and measure **Spearman
rank correlation against L2**. The evolution phase is unblocked only if some
cheap rung reaches **ρ ≳ 0.7**. Candidates are built by ablating and
recombining `token_features.py`'s named feature groups, **plus 3–4
deliberately-bad anchors** (random routing, single-feature routing,
byte-length-only): without a known-bad low end the correlation is measured over
a restricted range and reads optimistically.

Until the gate passes, `evolution-phase` is limited to pre-registration — the
problem directory, `metrics.yaml`, `task_description.txt`, an initial program,
and an explicit choice of MAP-Elites behavioural descriptors.

## Consequences

- ~4 GPU-h buys the right to spend, or the right to not spend, ~173 GPU-h.
- A negative result (no rung correlates) kills the evolution phase cheaply and
  is an accepted outcome, consistent with ADR 0001.
- `ladder-calibration` depends on `stage1-tree`'s modules: L0 scores a
  partition, so `tree_embed.py` and `token_features.py` must exist first. It
  does not depend on the stage-1 grid and can run beside it.
- MAP-Elites behavioural descriptors for a feature-extractor genome are
  currently unspecified anywhere. Writing the problem directory forces that
  choice, which is a second reason to do it before launching.
