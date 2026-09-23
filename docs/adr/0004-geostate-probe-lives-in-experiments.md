---
status: proposed
date: 2026-09-24
---

# 4. The GeoState decisive probe lives in `experiments/geostate/`; a standalone repository is a go-gate deliverable

Origin: unattended deep interview (`--auto` after Round 1); the decision is the
interviewer's recommendation, not a confirmed requirement.

## Context

GeoState (spec v0.1, 2026-09-23) proposes a new modality — one exactly
computed, invariant state token per geometric object — and a two-week decisive
probe (E0 consistency benchmark, V1 prototype against a matched baseline). The
obvious home is a fresh repository: the idea is unrelated to GigaEvo's
evolutionary search, its dependency profile (CGA kernel, peft, a training
loop) is not GigaEvo's, and a paper-oriented project eventually wants a clean
release.

Three facts in this repo cut the other way. No `experiments/` commit has ever
reached `origin/main` of the GitHub fork — the "repository" for research code
is a local arkhip workdir with CI, Redis, MLS staging and tests already wired.
`experiments/` already hosts two programmes with nothing in common
(`vocab_factorization`, `noregeo_synth`), so a third is the established
pattern. And the probe's expensive plumbing — the endpoint client with retries
and TLS, the MLS launcher pattern, the NoReGeo dataset/render/verify helpers,
HTML reporting — is here, while the probe may well return no-go.

## Decision

The probe is built as a **self-contained package `experiments/geostate/`** in
this repository. It may import from `experiments.noregeo_synth` only
`credentials`, `dataset`, `configs`, `figures` and `verify`; the retry loop of
`recovery_eval.request` is copied rather than imported; nothing is imported
from `gigaevo`. A test enforces the allowlist. The only change to GigaEvo's
`pyproject` is an optional extra `geostate`.

**Extraction into a standalone repository is the first task after a go
decision**, not a first-day task. Because the package is self-contained, the
extraction is a subtree split plus copying at most five helper modules.

## Considered options

- **New repository from day one.** Clean, but re-onboards arkhip (CLAUDE.md,
  Redis, MLS staging, tests) and duplicates the endpoint/MLS plumbing before
  the idea has survived its own kill gates.
- **Inside `noregeo_synth`.** Rejected: GeoState executes no model-written
  code, so it would inherit the guard/subprocess machinery for nothing, and
  the two programmes' "task" concepts differ (construction problem vs NoReGeo
  row).

## Consequences

- The training stack must never be pinned in GigaEvo's core dependencies; if
  that becomes necessary, that is the trigger to extract early.
- GeoState code follows this repo's conventions (configs-as-code in
  `configs.py`, plan `.md` format, tests under `tests/`).
- If the probe returns no-go, the package stays as a recorded negative result
  next to the plan, with no orphaned repository to maintain.
