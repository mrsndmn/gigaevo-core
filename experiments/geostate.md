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

Before any encoder is trained, two experiments can each kill the idea:

- **H0 (consistency).** Existing models are *inconsistent* under
  orientation-preserving similarities and relabelling: the same construction
  problem, moved, rotated, rescaled or with its points renamed, gets different
  constructions and different success. If the best open model is consistent
  at or above the kill gate on problems of five or more moves, the motivation
  for an equivariant state is gone.
- **H1 (state tokens vs text state).** After the same supervised training on
  the same trajectories, a model that receives the state as invariant tokens
  beats a *matched-training baseline* that receives the same state as a text
  table, by at least the go margin on problems of five or more moves. If the
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
- `generator.py` — seeded construction problems per length bucket.
- `aux.py` — NoReGeo point stems as construction problems (`--check`
  validates every row against its golden point).
- `dsl.py` — the language the model writes; parsed and executed by the
  kernel, never evaluated as code.
- `consistency.py` — E0: copies per problem, endpoint requests with cached
  retries, kernel scoring, agreement in the original frame, bootstrap CI,
  kill-gate verdict.

```bash
python -m experiments.geostate.generator --seed 1 --per-bucket 5 --show
python -m experiments.geostate.aux --check
python -m experiments.geostate.consistency --config e0-synthetic-text --dry
python -m experiments.geostate.consistency --config e0-synthetic-text --run
python -m experiments.geostate.consistency --config e0-synthetic-text --summary
python -m pytest tests/geostate -q
```

Outputs land in `run/geostate/<config>/` (gitignored): frozen
`problems.jsonl` and `copies.jsonl`, one directory per request with the
manifest, raw response and `eval.json`, and `summary.{json,md}`.

H1 (the V1 arm, its matched baseline, the MLS launcher and the training loop)
is **not implemented yet**; its gates and protocol are fixed in `configs.py`
and ADR 0006 so that they cannot drift once the numbers exist.

## Results

2026-09-24, build and offline verification only; no model has been queried.

- Kernel: null points, distance inner products, incidence, all three meet
  cases, parallel detection, translator/rotor/dilator/reflection and the
  similarity inverse agree with analytic geometry to float precision.
  Numerical equivariance of every primitive — including the order of an
  intersection pair — holds on random states and random group elements
  (`test_transition_is_equivariant`, 20 seeds).
- Auxiliary set: the kernel reproduces the golden point on **684 of 700**
  rows of the seven stems. The 16 skips are all degeneracies under the
  environment's own tolerances (thin triangles whose incentre construction
  meets a point within 2% of the diameter, two near-parallel line pairs whose
  angle is below the threshold, intersections that *are* a given point, a
  point lying on its own mirror axis), listed by `aux --check`.
- Generator: 300 problems (100 per length bucket) in 8 s; each trajectory
  reaches its target exactly at the last move and never before.
- E0 offline: with the reference trajectories standing in for the model,
  every copy succeeds and pairwise agreement is 1.0; the reference program of
  the original fails on a relabelled copy, so relabelling is a real test.
- Dry runs: `e0-synthetic-text` = 300 problems × 8 copies × 3 models = 7200
  requests; `e0-aux-text` = 140 × 8 × 3 = 3360; the figure arms one model each.

## Conclusions

The environment and the E0 harness are ready to run against the endpoint; the
first live measurement is `e0-synthetic-text --run --limit 10` to check the
prompt, then the full grid. The H0 verdict is read from `--summary` on the
"≥ 5 moves" row against the gate in `configs.py`.

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
  NoReGeo point rows; dry runs of all four E0 configs.
