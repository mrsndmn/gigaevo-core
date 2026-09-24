# GeoState: decisive probe for an exact, equivariant construction state

## Hypothesis

Multimodal and text LLMs lose track of *which geometric objects exist* while
building a construction: GeoBuildBench's most common failure is a reference to
an object that was never defined, and rendered feedback made it worse. GeoState
(spec v0.1, 2026-09-23; interview spec
`run/deep-interview/deep-interview-geostate-decomposition.md`) proposes to
keep the construction state outside the model's text — one exactly computed
token per object, carrying only quantities invariant under the plane's
similarities — so that the model's decisions cannot depend on the pose of the
figure and cannot name an object that does not exist.

Before any encoder is trained, two experiments can each kill the idea (the
length threshold, the kill margin and the applicability conditions of both
gates are constants in `configs.py`):

- **H0 (consistency).** Existing models are *inconsistent* under
  orientation-preserving similarities and relabelling: the same construction
  problem, moved, rotated, rescaled or with its points renamed, gets different
  constructions and different success. If the best open model is consistent
  at or above the kill gate on the long-construction bucket, the motivation
  for an equivariant state is gone. The verdict is withheld — not read — when
  copies are missing, when most copies produce no parseable answer, or when
  the model solves too few untransformed problems: a run with an expired
  token or a model that cannot use the language is "consistent" for the wrong
  reason.
- **H1 (state tokens vs text state).** After the same supervised training on
  the same trajectories, a model that receives the state as invariant tokens
  beats a *matched-training baseline* that receives the same state as a text
  table, by at least the go margin on the long-construction bucket. If the
  text table closes the gap, the contribution narrows to token economy and
  out-of-distribution poses (ADR 0006).

The unit of work is a **construction problem**: labelled given points, a
target described as a nested classical construction (midpoint, circumcentre,
reflection, ...), and success = the constructed object coincides with the
target under the exact kernel. NoReGeo's seven point-answer stems are recast
as such problems and used only as an auxiliary consistency set; their golden
answers validate the kernel and never enter a prompt.

## Setup

Code: `experiments/geostate/` (ADR 0004: a self-contained package next to
`noregeo_synth`, importing only its endpoint credentials, dataset, figure
renderer, scorer and config). The grid, the symmetry group, every tolerance
and both kill gates live in `configs.py`; nothing is repeated here.

- `cga.py` — Cl(3,1) conformal geometric algebra in numpy: points, lines,
  circles as IPNS vectors, meet by wedge + dual, versors for the similarity
  group. Validated against analytic geometry in `tests/geostate/test_cga.py`.
- `state.py` — the geometric state and the three primitive moves; exact,
  append-only; intersection pairs ordered by orientation (ADR 0005); typed
  move errors (`undefined_reference`, `type`, `degenerate`).
- `macros.py` — classical constructions expanded into primitives; the
  expansion *is* the target and the reference trajectory.
- `generator.py` — seeded construction problems per length bucket; moves the
  target does not depend on are pruned, and rejection counts are reported.
  Goals are coincidence with a constructed target, or — for a share of each
  bucket fixed in `configs.py` — a predicate with several valid answers: a
  point lying on two named curves (either intersection) or a line through a
  named point tangent to a named circle (either tangent). The points a curve
  or a tangent refers to may themselves be constructions, which is what
  spreads predicate goals over the length buckets; a tangent needs ten
  primitive moves, so it only appears in the long bucket. Problems solvable
  in one move over the given points, and descriptions naming constructions
  the reference never needs, are rejected. `goal_holds` is the single
  success check for E0 and V1, and a copy of a problem is accepted iff its
  reference answer mapped back still solves the original.
- `aux.py` — NoReGeo point stems as construction problems (`--check`
  validates every non-degenerate row against its golden point and exits
  non-zero on a mismatch).
- `dsl.py` — the language the model writes; parsed and executed by the
  kernel, never evaluated as code.
- `consistency.py` — E0: copies per problem (a copy whose target drifts from
  the original's, or duplicates another copy, is resampled), endpoint requests
  with cached retries, kernel scoring, agreement over *answered* copies in the
  original frame, Δsuccess and consistency with bootstrap CI, a breakdown by
  transform kind, and the kill-gate verdict on the gated config only. Failed
  requests are never scored; `--recover-errors` archives them for a re-run.

```bash
python -m experiments.geostate.generator --seed 1 --per-bucket 5 --show
python -m experiments.geostate.aux --check
python -m experiments.geostate.consistency --config e0-synthetic-text --dry
python -m experiments.geostate.consistency --config e0-synthetic-text --run
python -m experiments.geostate.consistency --config e0-synthetic-text --summary
python -m experiments.geostate.consistency --config e0-synthetic-text --recover-errors
python -m experiments.geostate.run_e0_jobs --config e0-synthetic-text [--limit 12] [--dry]
python -m pytest tests/geostate -q
```

Full grids run as MLS jobs (`run_e0_jobs.py`, one config per job, outputs under
`configs.ARTIFACT_ROOT`); local `--run` is for smokes. The endpoint omits its
intermediate TLS certificate, so the client verifies against a bundle that
adds the public GlobalSign intermediate (`certs/`).

Outputs land in `run/geostate/<config>/` (gitignored): frozen
`problems.jsonl` and `copies.jsonl` with a config fingerprint, one directory
per request with the manifest, raw response and `eval.json`, and
`summary.{json,md}`.

Reflection is the out-of-distribution condition of ADR 0005: the
`e0-synthetic-text-reflect` config reflects each problem and keeps only the
copies whose target is the reflection of the original target, which excludes
orientation-dependent constructions (quarter turns) by construction.

H1 — the V1 arm against its matched-training baseline (ADR 0006):

- `features.py` — exact invariant features of a new object against every
  existing one (kinds, scale-free distances, signed point-line offsets, line
  angles, incidence/tangency flags); no coordinate enters, so a state token is
  invariant by construction.
- `sequence.py` — one teacher-forced format for both arms: prompt, then the
  model's moves (the only positions with loss) alternating with
  environment-written state blocks; the `tokens` arm puts one `<st>` per
  object, the `table` arm the object's parameters as text.
- `encoder.py` / `v1_model.py` — the invariant-feature encoder whose output
  replaces the `<st>` embeddings, and a frozen Qwen3-1.7B with a hand-rolled
  LoRA applied identically in both arms.
- `train_v1.py` — generator trajectories held out by seed, LoRA-SFT, then
  the rollout evaluation; `rollout.py` — greedy decoding constrained to the
  legal moves of the current state (a reference to a missing object is
  impossible in either arm), kernel execution after every move, success /
  target-reached / wrong-selection / tokens per episode.
- `run_v1_jobs.py` — 2 arms × 2 seeds on MLS; `v1_report.py` — the table and
  the H1 gate (paired Δsuccess on the long bucket with bootstrap CI).

```bash
python -m experiments.geostate.train_v1 --arm tokens --seed 1 --out DIR --train --eval [--smoke]
python -m experiments.geostate.run_v1_jobs [--dry] [--arm tokens] [--seed 1]
python -m experiments.geostate.v1_report [--root DIR]
```

## Results

2026-09-24, build and offline verification only; no model has been queried.

- Kernel: null points, distance inner products, incidence, all three meet
  cases, parallel detection, translator/rotor/dilator/reflection and the
  similarity inverse agree with analytic geometry to float precision.
  Numerical equivariance of every primitive — including the order of an
  intersection pair — holds on random states and random group elements
  (`test_transition_is_equivariant`, 20 seeds). The adversarial review
  independently confirmed positions and pair order on 3190 random meets.
- Auxiliary set: the kernel reproduces the golden point on **684 of 700**
  rows of the seven stems with zero mismatches. The 16 skips are all
  degeneracies under the environment's own tolerances (thin triangles whose
  incentre construction passes within the minimum separation of a vertex,
  near-parallel line pairs below the angle threshold, intersections that *are*
  a given point, a point lying on its own mirror axis), listed by
  `aux --check`.
- Generator: 300 problems (100 per length bucket) in 8 s; each trajectory
  reaches its target exactly at the last move and never before, and contains
  no move the target does not depend on.
- E0 offline: with the reference trajectories standing in for the model,
  every copy succeeds and pairwise agreement is 1.0; the reference program of
  the original fails on a relabelled copy, so relabelling is a real test.
- Dry runs: `e0-synthetic-text` = 300 problems × 8 copies × 3 models = 7200
  requests; `e0-aux-text` = 684 × 8 × 3 = 16416; the figure arms one model
  each; `e0-synthetic-text-reflect` keeps 1011 of 1200 copies.
- Adversarial review of the first build found the harness scoring failed and
  unanswered copies as agreement (which could trip the gate with no data),
  an unrecoverable error cache, a unitless line comparison, lost copies in the
  per-transform breakdown, verdicts on ungated configs and dead moves in
  trajectories. All fixed and covered by tests.

2026-09-24, first live requests (smoke, 2 synthetic problems, local run):

- Every request failed with `APIConnectionError` until the client verified
  TLS against a bundle with the endpoint's missing GlobalSign intermediate;
  after that Qwen3.8-27B answered 12 of 12 requests with well-formed programs
  in 26–57 s, then the endpoint **unloaded the model mid-run** (404 "model not
  found"); Qwen3.5-27B and Qwen3-Next were already gone. Served since:
  Qwen3-8B, Gpt-oss-120b, two Medgemmas — so the figure arm has no
  figure-reading model today and the text arm runs on Qwen3-8B and
  Gpt-oss-120b, with Qwen3.8-27B picked up automatically if it returns.
- First inconsistency observed, on a 3-move problem ("the point symmetric to
  B with respect to A"): Qwen3.8-27B solved the identity, relabel, translate,
  rotate, scale and similarity copies and failed the similarity+relabel copy
  by naming the second point of the pair instead of the first — a selection
  error, the failure class H1 is about, not a syntax or reference error.
- Gpt-oss-120b answers in ~10 s and picks the wrong point of the pair
  consistently (1 of 8 copies right); Qwen3-8B thinks for 2.5–5k tokens and
  exhausted a 6144-token cap once. The selection rule now carries a worked
  example in the prompt and the cap is 16k (`configs.py`).

2026-09-24, E0 launched on MLS:

- `e0-synthetic-text` (gated config; models Qwen3-8B, Gpt-oss-120b, and
  Qwen3.8-27B if it returns): job `lm-mpi-job-83381a84-a9f7-4598-be37-a63d91b8ced7`,
  staged commit `0e575d89967a1dd3e595754e44d6da05feba4b5e`, artifacts
  `/mnt/virtual_ai0001053-00054_SR004-nfs2/d.tarasov/gigaevo-artifacts/results/geostate/e0-synthetic-text-7db7de275f91`
  (the suffix is the protocol hash). 7200 requests at concurrency 4; the
  endpoint token expires ~15 h after launch, so a resubmission after renewal
  is expected — the job script archives failed requests at the end and the
  launcher's dedup makes resubmission idempotent. No numbers yet.

2026-09-24, H1 built and launched:

- Local GPU smoke (20 optimiser steps on 36 trajectories, 6 rollouts per
  arm): both arms train (loss 1.19 → 0.07) and roll out end to end; the
  tokens arm solved 3 of 6, the table arm 2 of 6 with one wrong selection —
  numbers from a smoke, not results.
- MLS jobs (staged commit `c9c65aec75923414d6d044343bbb6818078c9e3f`, artifacts
  under `.../gigaevo-artifacts/results/geostate/v1/<arm>-s<seed>-f6181a88e7c2`):
  tokens s1 `lm-mpi-job-86aea567-79d3-4a62-a4fa-e86b37235cf4`,
  tokens s2 `lm-mpi-job-13fe2584-2650-49e4-9007-b49b6cad4c70`,
  table s1 `lm-mpi-job-fe162024-04b5-4a8d-b50c-f385635f2ddc`,
  table s2 `lm-mpi-job-dca181b8-a744-48ea-b151-97d00a569297`.
  The first job to start generates the shared data (~50k train, 900 eval)
  under a file lock; the others wait for it. No numbers yet.
- Review of the V1 commit (nothing that crashes these jobs or invalidates
  their comparison on a clean first run) led to protocol B, resubmitted as
  four more jobs: the LoRA init is now seeded before the model is built (in
  protocol A each run's init was unseeded, so the arms of one seed do not
  share it); a resubmitted job reuses `adapters.pt` instead of retraining
  into an old `eval.jsonl`; the keep set is decided on the table arm's
  lengths for both arms; gradients are clipped per parameter group; the
  protocol hash covers every module on the path and the report only reads
  runs of one protocol, withholds the verdict on missing or incomplete runs
  and carries bootstrap CIs per bucket; and — the substantive change — the
  encoder's pair features now carry the partner's relative index, so a state
  token can say *which* existing object it lies on, not only that it lies on
  one. Protocol A's verdict, if any, is a verdict on the index-free pooled
  encoder; protocol B is the one the plan's H1 reads.
- Protocol B jobs (commit `e24f16b6`, hash `89abc7561426`): tokens s1
  `lm-mpi-job-e69adec0-8628-4b88-873c-a82db63877e3`, tokens s2
  `lm-mpi-job-04c9416d-13bc-4e8f-bdfd-7c07815cfd66`, table s1
  `lm-mpi-job-b326e6a7-89bd-4b0a-9f36-4fff962b4718`, table s2
  `lm-mpi-job-53f4cf94-8d1a-4550-a091-6a646533dc02`. Read with
  `python -m experiments.geostate.v1_report --protocol 89abc7561426` (or
  `--protocol f6181a88e7c2` for protocol A).
- E0 auxiliary set launched too: `lm-mpi-job-75c1ee3d-1ea2-4fd0-a9c8-731458205e7e`
  (`e0-aux-text-7db7de275f91`).

2026-09-24 14:40, **V1 finished** (all eight jobs completed; training
1.3 h per tokens run and 1.7 h per table run, 900 rollouts each). Read with
`v1_report --protocol 89abc7561426` (B) and `--protocol f6181a88e7c2` (A);
reports in `run/geostate/v1-reports/`.

| protocol | arm | success 1-3 | success 4-9 | success 10+ | success ≥5 moves | wrong selection ≥5 | tokens/episode ≥5 |
|---|---|---|---|---|---|---|---|
| B | tokens | 0.828 | 0.805 | 0.637 | 0.709 [0.671, 0.742] | 0.032 | 388 |
| B | table | 0.947 | 0.940 | 0.873 | 0.903 [0.881, 0.926] | 0.008 | 627 |
| A | tokens | 0.883 | 0.880 | 0.745 | 0.806 [0.780, 0.831] | 0.013 | 396 |
| A | table | 0.947 | 0.940 | 0.888 | 0.913 [0.890, 0.934] | 0.006 | 627 |

Paired Δsuccess tokens − table on the ≥5-move bucket (544 problems, 2 seeds):
protocol B **−19.4 pts, CI95 [−22.8, −16.0] → KILL**; protocol A −10.7 pts,
CI95 [−13.1, −8.3] (the report withholds A's verdict only because its runs
predate `run.json`; all four are complete at 900/900). The state-token arm
loses to the same LoRA reading the same state as text on every bucket, in
both encoders, and the gap widens with construction length; its
wrong-selection rate on short problems (0.13 in B) says the token does not
convey *which* intersection is which as well as two rounded coordinates do.
Protocol B's partner-index encoder is worse than A's index-free one
(0.709 vs 0.806, CIs disjoint). What does hold is H3: the tokens arm
processes 38% fewer tokens per episode (388 vs 627) — at that accuracy cost.

## Conclusions

**H1 is killed under the pre-registered gate (ADR 0006):** with identical
data, LoRA, optimiser and constrained decoding, invariant state tokens are
10–19 points *behind* the textual state table on long constructions, not 10
ahead. Two readings, both to be checked before the idea is buried: (1) the
probe's encoder is a small invariant-feature MLP trained from scratch in one
epoch next to a frozen LM — a richer or pretrained encoder, more epochs, or a
learned (CGENN-style) encoder could close part of the gap, which the spec
listed as E2 ablations; (2) the text table carries exact rounded coordinates
and the tokens carry only relative invariants, so on the *in-distribution*
poses this probe tests, coordinates are simply more information. The spec's
remaining argument for tokens is out-of-distribution poses (H4) and token
economy (H3) — H3 is confirmed (−38% tokens), H4 is untested here.

E0 (H0) is still running; its verdict is read from `--summary` of the gated
config against the gate in `configs.py`.

Deviations from the interview spec, recorded rather than hidden: no
`geostate` extra was needed because the kernel is numpy-only and analytic
geometry, not kingdon, is the test oracle (AC-4); the auxiliary text config
covers every valid row while the auxiliary figure config samples per stem
(AC-11); incidence and tangency goals exist since the last commit (AC-10) but
the E0 and V1 runs launched before it use coincidence goals only — a
problem set with predicate goals is a new protocol hash, not a resume; the
gated models are no longer the spec's Qwen3.8-27B / Qwen3.5-27B / Qwen3-Next
but whatever open text models the endpoint serves — today Qwen3-8B and
Gpt-oss-120b — because the endpoint removed the others (E0 constraints).
While Qwen3.8-27B is absent the figure arm cannot run at all.

Next: (1) run E0 on the three open text models and the one figure model;
(2) build the V1 arm and its matched baseline on Qwen3-1.7B with generator
trajectories, as MLS `a100.1gpu` jobs; (3) apply the go/no-go rule from the
interview spec — both probes negative ⇒ stop; either positive ⇒ extract the
package into its own repository as the first post-go task (ADR 0004).

## Changelog

- 2026-09-24 — deep interview (11 rounds, unattended after round 1) fixed the
  scope: construction problems as the unit, `experiments/geostate/` as the
  home, orientation-preserving similarities as the group, matched-training
  baseline and numeric gates. Glossary terms in `CONTEXT.md`, ADRs 0004–0006.
- 2026-09-24 — kernel, state, macros, generator, auxiliary set, DSL executor
  and E0 harness implemented with 78 tests; kernel validated on 684/700
  NoReGeo point rows; dry runs of all E0 configs.
- 2026-09-24 — adversarial review: agreement only over answered copies,
  verdict withheld on incomplete or trivial data and printed only for the
  gated config, `--recover-errors`, angle-aware line comparison, per-kind
  breakdown keeps every copy, Δsuccess with CI, trajectory pruning, drift and
  duplicate rejection for copies, reflection OOD config, fingerprinted run
  directories, stratified `--limit`. 91 tests.
- 2026-09-24 — second review pass: a model with no data or missing copies
  now withholds the verdict (a single consistent model still falsifies);
  models excluded for the answered/success conditions are named; recovery
  also archives interrupted requests and never re-archives; the run
  fingerprint covers generator and tolerance settings; a macro node that adds
  no move is rejected so no description restates a sub-result. 96 tests.
- 2026-09-24 — V1 results: all eight runs complete; H1 KILL under the
  pre-registered gate (protocol B −19.4 pts, A −10.7 pts on ≥5 moves); H3
  (token economy) confirmed.
- 2026-09-24 — predicate goals (AC-10): incidence and tangency goals with
  several valid answers, one `goal_holds` for E0 and V1, protocol hashes
  cover the problem-generating code, launcher dedup keys on the protocol.
  Review pass: per-bucket goal quotas with reserved slots and nested curve
  points (incidence goals in every bucket), a one-move-shortcut guard, copies
  validated through the goal (reflection keeps predicate goals), `--summary`
  reads frozen inputs of any protocol. Generating 300 problems now takes
  ~35 s instead of 8 s. 129 tests.
- 2026-09-24 — V1 implemented (features, sequences, encoder, LoRA model,
  training, constrained rollouts, launcher, report), smoked on the local GPU
  and submitted as four MLS jobs; E0 auxiliary set submitted. 116 tests.
- 2026-09-24 — first live smoke: TLS bundle with the GlobalSign intermediate,
  MLS launcher `run_e0_jobs.py`, served-model check at run start, roster
  switched to Qwen3-8B + Gpt-oss-120b (+ Qwen3.8-27B when present), prompt
  example for pair selection, 16k token cap, concurrency 4. 100 tests.
