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
- **H4 (robustness to poses outside the training range).** Added
  2026-09-30 after the H1 kill, before any OOD episode existed. The *same*
  trained V1 adapters, with no retraining, on the frozen V1 evaluation set
  moved outside the training range (translated by 100, scaled ×10): in both
  arms the prompt's given coordinates leave the range the LoRA saw; in the
  table arm every state row does too, while the tokens arm's state blocks do
  not move. The statistic is retention on a common solved set, paired per
  problem: of the long problems both arms solve at identity in every seed,
  the share the tokens arm still solves under the pose minus the share the
  table arm does — not the absolute gap (it starts ~19 pts in the table's
  favour) and not raw point drops (they credit the tokens whenever the gap
  narrows, including when both arms collapse). H4 holds if on some far
  condition the tokens are more robust by at least the go margin with CI
  above zero; it is dead if every far condition is at/below the kill margin
  (`configs.H4_GATE`, the same margins as H1); withheld if identity does not
  reproduce the saved V1 episodes. Even a GO only argues against a
  *raw-coordinate* table: a table in a canonical frame would be invariant
  too, and that is the baseline a GO would have to face next; and a GO on
  robustness makes the tokens better in absolute terms only where the
  absolute gap also turns positive (both are reported).

The unit of work is a **construction problem**: labelled given points, a
target described as a nested classical construction (midpoint, circumcentre,
reflection, ...), and success = the constructed object coincides with the
target under the exact kernel. NoReGeo's seven point-answer stems are recast
as such problems and used only as an auxiliary consistency set; their golden
answers validate the kernel and never enter a prompt.

## Setup

Local figure/text paired cohorts: `run_figure_short50.main` selects frozen
text problems and their exact transformed copies; cohort selection and run
settings live in `configs.py`. The second cohort covers the remaining short
synthetic problems without overlapping the initial pilot.

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

H4 — `v1_ood.py` re-rolls the protocol-B adapters on each pose condition
(`configs.POSE_CONDITIONS`; poses built by E0's `transformed_problem`, ids
kept so episodes pair across arms, seeds and conditions); `identity` must
reproduce the saved V1 episodes, `rotate` makes coordinates non-integer at
about the training magnitude (458 of 883 rotated problems have a given point
just outside [-10, 10], up to ±14.1 — a near control, not a clean in-range
one), `scale-down` keeps the given
points inside the box (reported, not gated: the table's fixed 2-decimal
print loses precision there), `reflect` is ADR 0005's OOD condition, where
the tokens are not invariant either. A posed problem that is no longer the
same construction (other reference length; other trajectory without
reflection) is dropped. `run_v1_ood_jobs.py` freezes the conditions and
submits 2 arms × 2 seeds (evaluation only).

```bash
python -m experiments.geostate.v1_ood --prepare --eval --arm tokens --seed 1 --smoke
python -m experiments.geostate.run_v1_ood_jobs [--dry]
python -m experiments.geostate.v1_ood --report
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

2026-09-24 15:30, **E0 partial read** (both jobs still running, Qwen3-8B
only; the other two models have not started). A first look showed a
measurement artefact: the DSL parser rejected unambiguous programs models
write unprompted — `meet(l, line(A, B))`, `answer meet(l1, l2)`,
`meet(c, l)[0]` — as syntax errors (35% of aux copies). The parser now
desugars them and `consistency --rescore` re-evaluates cached outputs offline
(the model side is untouched). A review of that parser caught a second
defect — `answer_point = ...` was read as an answer line — fixed and rescored
again; records now carry a scorer hash and the summary warns when versions
mix, and the job-written summaries are stale until `--rescore` then
`--summary` are run after the jobs finish. After rescoring (16:50):

| set | bucket | problems | answered | success identity | success transformed | consistency [CI95] |
|---|---|---|---|---|---|---|
| synthetic | 1-3 (complete) | 100 | 0.837 | 0.596 | 0.565 | 0.627 [0.571, 0.678] |
| synthetic | ≥5 moves | 6 of 187 | — | — | — | (no data yet) |
| aux | 1-3 | 91 of 242 | 0.990 | 0.989 | 0.991 | 1.000 |
| aux | 4-9 (all ≥5 moves) | 130 of 442 | 0.905 | 0.769 | 0.781 | 0.833 [0.776, 0.887] |

What the partial data already says: Qwen3-8B is measurably inconsistent —
37% of answered copy pairs disagree on 3-move synthetic problems, 17% on
9-move NoReGeo constructions — and it cannot reach the 0.95 kill level on
the remaining aux problems even if every one of them were perfectly
consistent (the bound is 0.952 and the remaining stems are the harder
incircle/circumcircle ones). Its remaining failure mode is genuine: 12–14%
of synthetic copies exhaust the 16k-token budget while thinking (`empty`).
What it cannot say: the gate reads the *best* open model, and Qwen3.8-27B
(7 of 8 copies right in the smoke) and Gpt-oss-120b have not been measured
at scale — the E0 verdict waits for Qwen3.8-27B.

2026-09-30, **E0 final state: truncated, no verdict.** Both jobs ended
"completed" on 2026-09-24 ~15:15Z — the minute the endpoint token expired
(15:11Z); `run.json` counts 6317 of 7200 (synthetic) and 14609 of 16416 (aux)
requests not made. Only Qwen3-8B was measured; Gpt-oss-120b and Qwen3.8-27B
(listed as served at the end of both runs) have no data. Rescored with the
current scorer (`a2ca14b0a803`) and summaries rebuilt: synthetic 1-3 moves
unchanged (consistency 0.627 [0.571, 0.678]); synthetic ≥5 moves 11 of 187
problems, answered 0.40, success(identity) 0.18 — below both applicability
conditions, verdict withheld; aux ≥5 moves 135 of 442 problems, answered
0.892, success(identity) 0.741, consistency 0.804 [0.743, 0.865]. The E0
protocol (`7db7de275f91`, coincidence goals) is no longer the current code's
(`65b7e70ea8ab`, predicate goals): a resubmission after token renewal starts a
new protocol directory rather than resuming this one.

2026-09-30, **E0 relaunched** under protocol `65b7e70ea8ab` (predicate goals)
with a token valid to 2026-10-30. The 2026-09-24 runs queued Qwen3-8B first
and never reached another model, so the roster is now ordered by what decides
the gate: Qwen3.8-27B, Gpt-oss-120b, Google/Gemma-4-31B-it (an open model the
endpoint began serving), Qwen3-8B last. Local smoke (one 3-move incidence
problem, 8 copies): Qwen3.8-27B, Gpt-oss-120b and Gemma-4 8/8 correct,
~51 s / 3 s / 5 s per request. Jobs (staged commit `d28040a4`):
`e0-synthetic-text` `lm-mpi-job-d498cf6a-08d8-4f5b-8131-fb1e23515d3a`,
`e0-aux-text` `lm-mpi-job-1b289181-1179-460d-86a7-26cc644916f2`; artifacts
`e0-{synthetic,aux}-text-65b7e70ea8ab` under `configs.ARTIFACT_ROOT`.

2026-09-30, **H4 smoke** (2 problems per bucket, tokens and table seed 1,
real protocol-B adapters, local GPU): all six conditions build and roll out;
the identity condition reproduces the saved V1 episodes move for move (6/6
per arm), so the adapters load correctly under the current code (only the
goal check changed since V1, and it is equivalent for coincidence goals).
Of the 900 frozen problems, 11 no longer build under the generator's newer
guards (10 name an unneeded construction, 1 is a one-move shortcut) and drop
out of every condition.

2026-09-30, **H4 review before launch** (adversarial review; no OOD episode
existed yet): the first gate compared tokens − table directly and would have
read KILL even with the tokens untouched and the table losing 20 pts, since
the table leads by ~19 at identity — the gate now reads paired robustness;
scale-down moved out of the gate (inside the box, print-precision confound);
13 rotated and 5 reflected problems became different constructions after
rounding (e.g. 3 → 20 moves) — posed problems must now be the same
construction; retries re-pose 6 and 2 of them, so rotate ends 6 problems
short of identity (3 long) and reflect 2 more; the ood hash covers the
tolerances, buckets and rounding; identity must reproduce the saved V1
episodes (≥ 0.95 same success per run) or the verdict is withheld; a torn
`eval.jsonl` line no longer swallows the next record on resume.

2026-09-30, **H4 launched** (ood hash `3fb8e5f67028`, staged commit
`2466c314`; artifacts `v1-ood-3fb8e5f67028` under `configs.ARTIFACT_ROOT`).
Conditions kept / rejected of 900: identity 889/11, rotate 883/17,
translate-far 889/11, scale-up 889/11, scale-down 889/11, reflect 705/195.
Jobs: tokens s1 `lm-mpi-job-ba36e901-1179-4fdf-8184-678714241d17`,
tokens s2 `lm-mpi-job-ba349eee-f31d-4d6d-bd87-498e9af2ad97`,
table s1 `lm-mpi-job-9614ef8e-502d-4624-b2bf-f96ca1576c44`,
table s2 `lm-mpi-job-622ce9d6-f190-424e-8815-49727232976f`.
Expected ~5 h each (~0.86 h per 900 rollouts in V1). Read with
`python -m experiments.geostate.v1_ood --report`.

2026-09-30, **H4 gate, second review pass** (report-only; the running jobs,
their frozen conditions and the ood hash are unchanged, and only identity
episodes existed — about 330 per run, all reproducing V1 move for move): the
first fix gated on raw drops, (tokens − tokens@identity) − (table −
table@identity), which is the far gap minus the identity gap — so any pose
that narrows the gap reads as tokens robustness, including both arms
collapsing (equal 52% proportional losses would read GO). The gate now reads
retention on the 327 long problems both arms solve at identity in every seed
(`H4_GATE` docstring). Identity reproduction also requires every fresh
episode to have a saved counterpart.

2026-09-30 ~21:15, **E0 resubmitted with 16 requests in flight per job**
(was 4; commit `194c9e2b`; `workers` is outside the protocol hash and the
fingerprint, so the same `-65b7e70ea8ab` directories resume). The first two
jobs were killed after 2.7 h with 363 synthetic and 1533 aux answers cached;
the 4 in-flight requests of each were archived by `--recover-errors` and
re-issued. New jobs: `e0-synthetic-text`
`lm-mpi-job-ac1d6bf5-33f5-438b-a849-70ac6954b1da`, `e0-aux-text`
`lm-mpi-job-43e7a568-53cc-4d8a-a8de-0d6329e89831`. First 20 min, no request
errors: synthetic ~6.8 answers/min (was ~2); aux is on the long
incircle/circumcircle stems (median 10k completion tokens, ~12 min per
request — per-request decoding halves at 32 in flight, total throughput per
job roughly doubles).

2026-10-01 00:17, **H4 finished — KILL.** All four eval-only jobs completed
(5144 rollouts each). Identity reproduces the saved V1 episodes on 889 of 889
problems per run, move for move, so these are H1's models. Report:
`v1_ood --report` → `v1-ood-3fb8e5f67028/report.md`.

| condition | tokens success ≥5 | table success ≥5 | tokens − table (abs) | retention tokens | retention table | robustness [CI95] |
|---|---|---|---|---|---|---|
| identity | 0.708 | 0.902 | −19.4 | — | — | — |
| rotate | 0.710 | 0.910 | −20.0 | 0.988 | 0.998 | −1.1 [−2.0, −0.3] |
| translate-far (gated) | 0.706 | 0.898 | −19.2 | 0.991 | 0.995 | −0.5 [−1.4, +0.3] |
| scale-up (gated) | 0.710 | 0.908 | −19.8 | 0.992 | 0.998 | −0.6 [−1.4, +0.0] |
| scale-down | 0.708 | 0.907 | −19.9 | 0.986 | 0.998 | −1.2 [−2.3, −0.3] |
| reflect | 0.682 | 0.915 | −23.3 | 0.880 | 1.000 | −12.0 [−16.1, −7.9] |

Retention is over the 327 long problems both arms solve at identity in every
seed (208 for reflect). The premise of H4 does not hold: the raw-coordinate
table does *not* break when the figure is moved 100 units away or scaled ×10
— it keeps 99.5–99.8% of what it solved — so there is no fragility for the
invariant state to fix. Neither arm moves under similarities (all Δ vs
identity within ±1 pt). Under reflection, the one transform that changes the
tokens (signed features flip), the tokens arm loses 12% of its solved
problems and the table loses none: the text coordinates carry orientation
through a mirror better than the learned encoder does.

2026-10-01 00:20, **E0 interim** (Qwen3.8-27B only so far, 16 workers, no
request errors; read in memory, the job's files untouched): synthetic 914 of
2393 copies — 1-3 moves consistency 0.989 [0.978, 0.998] (Qwen3-8B had
0.627), ≥5 moves only 12 problems yet (answered 0.72: long problems often
exhaust the 16k budget, which would withhold the gate on the applicability
condition); aux ≥5 moves (130 of 442 problems) consistency 0.957 [0.935,
0.977], above the 0.95 kill level on the auxiliary set (not gated).

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

**H4 is killed too (2026-10-01):** the trained text-table baseline is
already pose-robust — moved 100 units or scaled ×10 it keeps 99.5–99.8% of
its solved long problems, the tokens arm 99.1–99.2% — so the out-of-range
fragility the invariant state was meant to remove does not exist for a
LoRA-tuned Qwen3-1.7B reading 2-decimal coordinates; under reflection the
tokens are the fragile arm (−12% retention vs 0%). With H1 and H4 both
negative, the state-token design has no measured advantage left except token
economy (H3, −38%), bought at −19 pts of success.

E0 (H0) is still running. Qwen3-8B (2026-09-24) was inconsistent — 37% of copy
pairs disagree on 1-3-move problems — but the stronger Qwen3.8-27B is
consistent at 0.989 on short synthetic problems and 0.957 on long NoReGeo
constructions so far, which points towards H0 being killed by the best open
model once its long synthetic bucket completes (or withheld, if too many long
copies exhaust the token budget).

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

Next: (1) let E0 finish and read the gate (`consistency --summary` on the
`-65b7e70ea8ab` directories); (2) apply the go/no-go rule from the interview
spec — with H1 and H4 negative, only a positive H0 keeps anything alive, and
H0 alone justifies the motivation, not the state-token design. The one
untested rescue is a stronger encoder (E2 ablations: more epochs, pretrained
or CGENN-style), which would now have to beat a baseline that is both more
accurate and already pose-robust.

## Changelog

- 2026-10-07 — added a second local paired figure cohort for the remaining
  short synthetic problems; separate artifacts and tmux execution.

- 2026-09-30 — E0 read after the jobs ended: truncated at token expiry,
  Qwen3-8B only, rescored; H0 verdict withheld.
- 2026-10-01 — H4 KILL: the text table is pose-robust (retention ≥ 99.5% under
  translate ×100 / scale ×10), tokens lose 12% under reflection; E0 interim:
  Qwen3.8-27B consistency 0.989 (short synthetic), 0.957 (long aux).
- 2026-09-30 — token renewed; E0 relaunched with the decisive models first
  (+ Gemma-4-31B-it); H4 added: `v1_ood.py` re-evaluates the V1 adapters
  under six pose conditions, `run_v1_ood_jobs.py`, gate `H4_GATE`; review
  pass before launch: robustness gate, same-construction check, reproduction
  gate, hash coverage, torn-line resume; second pass: retention on the common
  solved set instead of raw drops, reproduction coverage. 140 tests.

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
- 2026-09-24 — DSL parser desugars nested calls, `answer <expr>` and
  `[k]` selection; `consistency --rescore [--force]` re-evaluates cached
  outputs; partial E0 rescored (aux ≥5-move answered 0.38 → 0.89). 130 tests.
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
