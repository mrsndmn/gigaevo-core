# GigaEvo — LLM-driven evolutionary program search

Evolutionary framework (MAP-Elites) that mutates Python programs with LLMs and
selects on task fitness. Upstream: <https://github.com/mrsndmn/gigaevo-core>.
Full docs in `README.md` and `docs/` (QUICKSTART, ARCHITECTURE, DAG_SYSTEM,
EVOLUTION_STRATEGIES, COEVOLUTION, memory).

This file is **untracked** (`.git/info/exclude`) — it is local arkhip context,
not part of the upstream repo. Same for `.arkhip/`.

## Environment

- Conda env: `/home/jovyan/.mlspace/envs/gigaevo-core` (Python 3.12), installed
  as `pip install -e ".[chains,plotting,optimization,tracking,dev,test]"`.
  The `local-llm` and `memory-platform` extras are **not** installed — add them
  only if a task actually needs the sudoku local runtime or the GAM platform
  backend. (`torch` is nonetheless present: `[chains]` pulls it in via
  `sentence-transformers`. `vllm` is not.)
- `mls` is **installed**, but in the `arkhip` env, not this one — binary at
  `/home/jovyan/.mlspace/envs/arkhip/bin/mls`, library importable only from
  `/home/jovyan/.mlspace/envs/arkhip/bin/python`. Evolution runs locally;
  `experiments/vocab_factorization` submits MLSpace jobs (see below).
  Subcommands work; `mls --help` does **not** (its formatter wants
  `click.decorators.HelpOption`, gone in the env's click 8.3.2) — read
  `mls <cmd> --help`-less usage from the source instead of trusting the crash.
- **Launchers run under `arkhip`, jobs run under `gigaevo-core`.** The MLSpace
  launchers need `mls`; the entrypoints they submit need `torch`/`transformers`.
  No single env has both, so `configs.py` is kept standard-library-only and the
  launchers import the grid from it:

  ```bash
  PYTHONPATH=. /home/jovyan/.mlspace/envs/arkhip/bin/python \
      -m experiments.vocab_factorization.run_train_jobs --dry
  ```

  Worker containers mount the workspace at `/workspace-SR004.nfs2/`, not
  `/mnt/virtual_ai0001053-00054_SR004-nfs2/`; `jobs.to_job_path` does the
  rewrite and `jobs.py` is the only place that knows about it. Artifacts land
  in `/workspace-SR004.nfs2/d.tarasov/gigaevo-artifacts/`.
- `GIGAEVO_REDIS_SOCKET` and `PYTHONPATH=.` are set as conda env vars, so an
  activated shell is already pointed at the right Redis.

## Redis (userspace, unix socket only)

Full spec — install, config rationale, key schema, DB allocation, call sites
that bypass the socket, runbook — in `.arkhip/redis.md`.

Redis 7.2.7 is built and installed under `~/.local` — no root, no system
service. It listens on a **unix domain socket only** (`port 0` in the config),
so access is a filesystem question rather than a password question: the socket
sits in a 0700 directory owned by this user, and nothing else on this shared
node can reach it. `requirepass` is therefore unnecessary — which matters
because gigaevo has no password support (see `gigaevo/utils/redis_conn.py`).

| Thing | Path |
|---|---|
| Launcher | `~/.local/bin/gigaevo-redis` |
| Binaries | `~/.local/bin/redis-{server,cli,benchmark}` |
| Config | `~/.config/redis/gigaevo.conf` |
| Socket | `~/.local/state/redis/gigaevo.sock` |
| RDB + pid + log | `~/.local/state/redis/` (mode 0700) |

```bash
gigaevo-redis start          # idempotent: a live server is left alone
gigaevo-redis status         # pid, PING, memory, dbsize
gigaevo-redis cli dbsize     # redis-cli bound to the socket
gigaevo-redis logs -f
gigaevo-redis stop
```

Because there is no TCP listener, any component that still builds a
`redis://host:port` URL fails immediately instead of silently reaching some
other user's server — a missed call site is loud, not silent.

## Running evolution

```bash
gigaevo-redis start
export GIGAEVO_REDIS_SOCKET=$HOME/.local/state/redis/gigaevo.sock   # set in the env already
python run.py problem.name=heilbron max_mutants=5
```

`redis.socket=` (Hydra) overrides `redis.host`/`redis.port` for anything
instantiated from `config/`; the env var additionally covers the call sites
that carry no connection config of their own (LLM endpoint pool, migration
bus, `tools/`). Prefer the env var. The CLI takes `--redis-socket`.

Parallel runs share one server and separate by **db index** and key prefix:
`python run.py problem.name=X redis.db=1`. `gigaevo-redis cli -n 1 dbsize`
inspects db 1.

LLM access needs `OPENAI_API_KEY` in `.env` (OpenRouter key by default).

## Usage arms

Every way the code in this repo is launched. A change to shared code has to
hold on all of the arms it touches, not just the one that motivated it.

| Arm | Command | Env |
|---|---|---|
| Evolution | `python run.py problem.name=<p> ...` (Hydra) | gigaevo-core |
| Tests | `python -m pytest -q -m "not slow"` | gigaevo-core |
| Tooling / reports | `python tools/<script>.py` | gigaevo-core |
| Vocab-factorization jobs | `PYTHONPATH=. .../arkhip/bin/python -m experiments.vocab_factorization.run_train_jobs --dry` | **arkhip** (needs `mls`) |
| Vocab-factorization compute | `python -m experiments.vocab_factorization.<prepare\|build_arms\|calibrate\|plot>` | gigaevo-core |
| NoReGeo synthesis | `python -m experiments.noregeo_synth.generate --task <stem>:<qid>` | gigaevo-core |
| NoReGeo VLM eval | `python -m experiments.noregeo_synth.eval_blind --config <c> --per-stem <n>` | gigaevo-core |
| GeoState probe (E0) | `python -m experiments.geostate.consistency --config <c> --dry\|--run\|--summary` | gigaevo-core |
| GeoState local paired figure cohorts | `python -m experiments.geostate.run_figure_short50 [--config <paired-config>] [--dry]` — freezes paired text problems and exact copies for short/medium/long/aux cohorts; local-only configs reject MLS submission | Python with numpy/openai/matplotlib |
| GeoState local figure queue | `python -u -m experiments.geostate.run_figure_queue` — serialized cohorts from `configs.FIGURE_QUEUE`, recover failed requests before each pass, cache successes, persist `figure-queue-state.json`, singleton controller lock | Python with numpy/openai/matplotlib; detached tmux |
| GeoState E0 jobs | `python -m experiments.geostate.run_e0_jobs --config <c> [--limit N] [--dry]` | gigaevo-core (calls arkhip `mls`) |
| GeoState V1 train/eval | `python -m experiments.geostate.train_v1 --arm tokens\|table --seed N --out DIR --train --eval [--smoke]` | gigaevo-core (GPU) |
| GeoState V1 jobs / report | `python -m experiments.geostate.run_v1_jobs [--dry]`, `python -m experiments.geostate.v1_report --protocol 89abc7561426 [--root DIR]` | gigaevo-core (calls arkhip `mls`) |
| GeoState H4 (V1 adapters, OOD poses) | `python -m experiments.geostate.v1_ood --prepare\|--eval --arm A --seed N\|--report [--smoke]`, `python -m experiments.geostate.run_v1_ood_jobs [--dry]` | gigaevo-core (GPU; launcher calls arkhip `mls`) |

### NoReGeo synthesis (`experiments/noregeo_synth/`)

Turns NoReGeo benchmark rows into step-by-step reasoning traces where every
step carries matplotlib code that adds that step's construction to a shared,
persistent canvas (one PNG per step). Plan: `experiments/noregeo-synth.md`;
grid and every tolerance: `experiments/noregeo_synth/configs.py`.

```bash
python -m experiments.noregeo_synth.generate --task outer_circle_center:6200
python -m experiments.noregeo_synth.generate --config qwen38-27b --n 20 --stem triangle_area
```

- Model-written code is executed. It is screened by an AST guard
  (`guard.py`) and then run in a separate `python -I` subprocess with a
  timeout — a name-level guard rail plus process isolation, not a sandbox.
  Keep it that way, and do not weaken the acceptance checks in `verify.py`
  without a test: `tests/noregeo_synth/` exists because every failure they
  catch is silent (a step that draws nothing still writes a PNG).
- Inference goes to the internal OpenAI-compatible endpoint
  `https://inference.airi.net/v1` (vLLM). `credentials.py` resolves the token
  from `AIRI_API_KEY`/`OPENAI_API_KEY`, else from `$AIRI_ENV_FILE`,
  `<repo>/.env`, `~/.config/airi/env`, `../graphify/.env` — where the JWT
  currently lives. The token expires (`resolve()` refuses an expired one), so
  a 401 mid-run means "reissue", not "wrong URL".
- Outputs land in `run/noregeo_synth/<config>/` and cached benchmark files in
  `data/noregeo/`; both are already gitignored.

### NoReGeo coordinate-blind eval (`eval_blind.py`)

Measures whether a VLM can solve a NoReGeo task when the coordinates are in the
figure rather than in the text, and — with the `swap` arm — whether it is
reading the figure at all. Plan: `experiments/noregeo-vlm-eval.md`; arms,
models and every tolerance: `configs.py`.

```bash
python -m experiments.noregeo_synth.eval_blind --config qwen38-27b --per-stem 5
python -m experiments.noregeo_synth.eval_blind --arm figure --arm swap --per-stem 20
python -m experiments.noregeo_synth.eval_blind --summary --by-stem
```

- The benchmark's `im_path` (`images_full/`) **prints the answer in the figure
  title**; only `im_dots_path` may be shown to a model, which is what
  `EVAL_IMAGE_FIELD` pins and what `tests/noregeo_synth/test_eval_blind.py`
  asserts. Treat any change there as a correctness bug, not a refactor.
- Not every model the endpoint accepts an image for can see it:
  `Openai/Gpt-oss-120b` answers "I can't see images", so it is a blind control
  (`--config gptoss-120b-blind`), never a data point.
- Outputs: `run/noregeo_eval/<config>/records.jsonl` plus the rendered swap
  figures next to it.

## Testing

```bash
python -m pytest -q -m "not slow"     # the arkhip testing-stage command
```

The suite uses `fakeredis`, so it does **not** need the server running.
`tests/integration/test_redis_unix_socket.py` is the exception: it skips unless
`GIGAEVO_REDIS_SOCKET` points at a live socket, so start Redis to exercise it.

## Conventions

- Ruff (`line-length 88`, `E,F,I,UP`) + mypy; `pre-commit` config is in the repo.
- Hydra owns all configuration — add knobs to `config/`, not as literals.
- New Redis connections go through `gigaevo.utils.redis_conn`
  (`build_redis_url` / `redis_connect_kwargs`), never a hand-built
  `redis://{host}:{port}/{db}` — that is what makes the socket switch total.

### NoReGeo intermediate feedback (`feedback.py`)

- Paired rollouts: `python -m experiments.noregeo_synth.feedback --task outer_circle_center:6200`
- Saved reports: `python -m experiments.noregeo_synth.feedback --summary --out <run directory>`
- Tests: `python -m pytest tests/noregeo_synth/test_feedback.py -q`
- Design: `feedback.py:experiment_configs`; plan: `experiments/noregeo-feedback.md`.
  Reuses synthesis prompts, endpoint calls, guarded subprocess execution and
  offline verification. Gold is never sent to the policy; both arms share the
  first action. `feedback_execute.DrawingSession` launches `_feedback_runner.py`
  under `python -I` and retains the process between actions, with per-action
  timeouts. Existing synthesis and static VLM evaluation remain usage arms
  of those shared helpers. Outputs go to timestamped `run/noregeo_feedback/`.

### NoReGeo HTML analysis

- `python -m experiments.noregeo_synth.feedback_report --run <run directory> --out <report.html>`
  renders saved paired rollouts as one offline HTML file with embedded PNGs.
  Repeat `--note` to add explicitly human analysis. Supports initial pilot
  `turn_XX/` frames and current `frames/` layout; never calls inference or code execution.
- API: `feedback_report.build_report(run, out, notes=())`.
- Tests: `python -m pytest tests/noregeo_synth/test_feedback_report.py -q`.
- Existing feedback, synthesis, static VLM evaluation and their CLIs are unchanged.

### NoReGeo batch feedback evaluation

- Design: `feedback_eval.py:experiment_configs`; plan `experiments/noregeo-feedback-eval.md`.
- Launch/dry: `python -m experiments.noregeo_synth.run_feedback_eval [--dry]`.
  Calls installed mls CLI as API adapter; stages committed code and reuses the
  existing out-of-workdir NFS artifacts root. No separate training launcher:
  this experiment is inference-only. Credentials remain in the shared env file.
- Worker: `python -m experiments.noregeo_synth.feedback_eval --run --out <root>`;
  `--prepare` freezes the inputs, `--report` rebuilds offline comparison HTML.
- APIs: `prepare`, `cohort`, `run`, `evaluate_task`, `comparison`; resume must
  preserve the common first action and failed-attempt artifacts.
- Tests: `python -m pytest tests/noregeo_synth -q`.
- Shared usage: `feedback.write_json` now writes records atomically so the
  concurrent batch reporter cannot read partial files. Pilot feedback CLI,
  HTML reports, old static eval and synthesis remain relevant usage arms.
- Batch failure semantics: drawing timeouts terminate that measurement as wrong;
  request errors remain retryable. Resume recovers shared actions from either
  arm, checks consistency, and the paired reporter excludes protocol violations.
  `--report` may run concurrently with the worker (unique atomic output files).

### NoReGeo recoverable feedback v2 — Usage arms

- `python -m experiments.noregeo_synth.run_recovery_eval [--dry]`: committed
  staging, external NFS outputs, MLS dedup and frozen inputs; old launchers unchanged.
- `python -m experiments.noregeo_synth.recovery_eval --prepare --out <root>`:
  freezes old failure contexts and outcome-blind pilot/cohort selection.
- `python -m experiments.noregeo_synth.recovery_eval --run --out <root>`:
  format diagnostics → technical gate → recoverable pilot → technical gate →
  full cohort → separate synthetic midpoint corruption. Gold is offline only.
- `python -m experiments.noregeo_synth.recovery_eval --report --out <root>`:
  standalone HTML from saved raw replies, execution errors and frames; no inference.
- `_recovery_runner.py` is a `python -I` subprocess only, transactional replay
  with stdout capture; failed code cannot affect later actions. Static guard
  remains a guard rail, not a security sandbox.
- `python -m pytest tests/noregeo_synth/test_recovery_eval.py -q` exercises
  parsing, rollback, equal observation arms, technical gates and offline reporting.
  Existing synthesis, feedback, old batch eval and their report/test arms continue
  to use their original entrypoints and configs.
- V2 policy environments allowlist input fields before building canvas bounds;
  `dataset.coordinate_pairs` intentionally includes gold for legacy synthesis,
  so never pass an unfiltered evaluation row into `build_preamble`.
- Explicit HTTP retry loop owns all retries (SDK retries disabled). An existing
  request manifest without result blocks reroll after interruption. Technical
  completion requires visible drawing plus non-null computed answer, not gold
  correctness. Old failure-context diagnostics preserve original prompts only.

### NoReGeo additional recovery cohort — Usage arms

- `python -m experiments.noregeo_synth.run_recovery_eval --experiment extra15 [--dry]`
  selects new registry/root/module; default original launcher behavior is preserved.
- `python -m experiments.noregeo_synth.recovery_extra --prepare|--run|--report --out <root>`
  selects tasks absent from the first pilot, freezes its records, evaluates the
  same recovery protocol and builds `comparison.html` plus embedded `analysis.html`.
- `python -m pytest -o addopts='' --confcutdir=tests/noregeo_synth tests/noregeo_synth/test_recovery_extra.py -q`.
- Shared usage arms: original launcher dry/submit and recovery pair/rollout/replay,
  report, old diagnostics, all NoReGeo tests. No changes to model/action settings.
- Extra-cohort TLS: `prepare` writes a root bundle plus the CA-issued GlobalSign
  intermediate to its external output directory; launcher passes SSL_CERT_FILE.
  Public PEM provenance: https://secure.globalsign.com/cacert/gsgccr3dvtlsca2020.crt;
  chain verified with openssl against system roots. No verify=False paths.
- `python -m experiments.noregeo_synth.recovery_extra --recover-connections --out <root>`
  is an explicit offline recovery action: refuses active jobs, non-connection
  errors, unfinished rollouts and responses with raw completions; archives all
  task artifacts then invalidates only exhausted connection failures. Successful
  cached replies and completed arms remain unchanged. Resubmit via normal launcher.

### NoReGeo latest-frame feedback — Usage arms

- `python -m experiments.noregeo_synth.run_recovery_eval --experiment last30 [--dry]`
  launches a third arm on the same30 tasks, reusing saved first responses and TLS bundle.
- `python -m experiments.noregeo_synth.recovery_last --prepare|--run|--report --out <root>`
  freezes the original60 records and builds a three-arm HTML comparison plus new traces.
- Shared `recovery_eval.context_for` runs at action and final-request boundaries:
  history/canvas unchanged, canvas_last retains all text and exactly the latest image
  (or none before first success). Failed actions retain last committed image;
  replacing a step updates it. No mutation of saved requests/full trace history.
- `python -m pytest -o addopts='' --confcutdir=tests/noregeo_synth tests/noregeo_synth/test_recovery_last.py -q`.
- Affected shared arms: original and extra15 recovery rollouts/reports/tests, all
  launcher choices, isolated replay unchanged. Old completed artifacts are read only.

### Full NoReGeo feedback — Usage arms

- `python -m experiments.noregeo_synth.run_recovery_eval --experiment full [--dry]`:
  independent full-benchmark registry, MLS staging/dedup and TLS bundle.
- `python -m experiments.noregeo_synth.recovery_full --prepare|--run|--report --out <root>`:
  freeze all tasks, bounded concurrent evaluation, offline HTML failure browser.
  Worker lock excludes duplicate direct launches. Re-submit after infrastructure
  stop to resume; received responses are reused, unknown in-flight calls block.
  Credentials are reloaded per request; renew their persistent env file if expired.
- Shared recovery rollout accepts optional request callback for durable transport
  recovery; original/extra15/last30 retain default callback and protocol.
- `python -m pytest -o addopts='' --confcutdir=tests/noregeo_synth tests/noregeo_synth -q`
  exercises old entrypoints, request context retention, replay and full recovery.
- Full reports read checkpointed records concurrently; primary scores exclude
  known dataset defects, all-task results retain them. Token counts per arm include
  the shared first response; physical accounting counts it once and can lag live calls.

### GeoState decisive probe — Usage arms

Plan: `experiments/geostate.md`; interview spec:
`run/deep-interview/deep-interview-geostate-decomposition.md`; decisions:
ADRs 0004–0006; glossary: `CONTEXT.md` ("Geometry / GeoState terms").
Grid, group, tolerances and kill gates: `experiments/geostate/configs.py`
(stdlib-only, like the other `configs.py`).

- `python -m experiments.geostate.generator --seed <s> --per-bucket <n> [--show] [--out f.jsonl]`
  — seeded construction problems with reference trajectories.
- `python -m experiments.geostate.aux --check` — recasts the seven NoReGeo
  point stems as construction problems; every non-degenerate row must
  reproduce its golden point (exit 1 on a mismatch; degenerate rows are
  listed and skipped). Needs the cached `data/noregeo/` task files.
- `python -m experiments.geostate.consistency --config <name> --dry` — freezes
  problems and copies under `run/geostate/<name>/` (fingerprinted: a directory
  prepared for another config is refused), no requests;
  `--run [--limit N] [--workers W]` queries the endpoint (credentials via
  `noregeo_synth.credentials`, rebuilt before the token expires; one
  request directory per copy, a lock file per run; `--limit` is taken across
  buckets). Successful responses are cached; a failed request is cached too
  and is never scored — `--recover-errors` archives failed directories so the
  next `--run` re-issues them. `--summary` writes `summary.{json,md}` and
  reads frozen inputs of *any* protocol (the header says which; a run under
  another protocol is refused, a summary is not); the kill-gate verdict
  appears only for the gated config and is withheld on incomplete or trivial
  data. Job descriptions and run directories carry the protocol hash; a code
  change is a new protocol, never a resume.
- Problems: coincidence targets plus predicate goals (a point on two named
  curves, a tangent through a point) with per-bucket quotas
  (`GeneratorSpec`); `generator.goal_holds` is the one success check.
  Generating 300 problems takes ~35 s (quotas + one-move-shortcut guard); the
  50k V1 training set is generated once per data key in the first job.
- `python -m experiments.geostate.run_e0_jobs --config <name> [--limit N] [--dry]`
  — stages the committed tree and submits one E0 config to MLS (same pattern
  as `noregeo_synth.run_recovery_eval`: commit-hash staging, external
  artifacts root `configs.ARTIFACT_ROOT`, credentials env file and TLS bundle
  via `SSL_CERT_FILE`, dedup on active job description). The job runs `--run`
  then `--summary`; the summary can be rebuilt locally from the NFS outputs
  with `consistency --summary --out <artifact dir>`.
- TLS: the endpoint omits its intermediate certificate; `client.make_client`
  verifies against certifi's roots plus the GlobalSign intermediate shipped in
  `experiments/geostate/certs/` (written once as `tls-ca-bundle.pem` next to
  the run), or against `SSL_CERT_FILE` when set. Without it every request
  fails with `APIConnectionError` — that is what the first live smoke showed.
- `python -m pytest tests/geostate -q` — kernel vs analytic geometry,
  equivariance of every primitive, generator determinism, DSL executor, offline
  E0, auxiliary golden reproduction, the launcher manifest and the import
  boundary (the launcher alone may import the vocab-factorization job plumbing).
- Boundary (ADR 0004, enforced by `tests/geostate/test_boundaries.py`): the
  package imports nothing from `gigaevo` and from `experiments.noregeo_synth`
  only `credentials`, `dataset`, `configs`, `figures`, `verify`. The retry loop
  is a copy of `recovery_eval.request`; changes there do not propagate. Model
  output is parsed by `dsl.py` and executed by the kernel — nothing the model
  writes is ever evaluated as Python, so the guard/subprocess machinery of
  `noregeo_synth` is deliberately not used.
- V1 (H1, ADR 0006): `train_v1.py --arm tokens|table --seed N --out DIR
  --train --eval [--smoke] [--data-root R]` trains a hand-rolled LoRA on a
  frozen Qwen3-1.7B (local HF cache, `HF_HOME`; no peft dependency) on
  generator trajectories held out by seed, then rolls out the eval set under
  constrained decoding (`rollout.py`: legal moves only, KV cache, kernel
  execution per move; resumable `eval.jsonl`). The two arms differ only in the
  state blocks (`sequence.py`): `<st>` tokens whose embeddings come from the
  invariant-feature encoder (`features.py`, `encoder.py`) vs the object
  parameters as text. `run_v1_jobs.py` submits 2 arms × 2 seeds to MLS
  (shared data under `ARTIFACT_ROOT/v1/data-<hash>`, prepared once under a
  file lock); `v1_report.py --protocol 89abc7561426 [--root DIR]` aggregates
  and reads the H1 gate (the default protocol is the *current* code's hash,
  which has moved on since V1 ran, so without `--protocol` it reads nothing).
  `--smoke` runs 20 steps on 36 trajectories and 6 rollouts on the local GPU
  (~2 min, ~15 GB). Tests use a tiny random Qwen3 on CPU with 4 torch threads
  (256 threads make tiny matmuls ~100x slower).
- H4 (`v1_ood.py`, no training): the V1 protocol-B adapters
  (`V1_OOD.source_protocol`) re-rolled on the frozen V1 eval set
  (`data-<V1_OOD.data_key>`) under each `configs.POSE_CONDITIONS` pose —
  identity (must reproduce the saved episodes, else the verdict is
  withheld), rotate, translate-far, scale-up, scale-down, reflect. Poses go
  through `consistency.transformed_problem` (shared with E0) and keep the
  original problem ids; a posed problem that changes reference length (or,
  without reflection, trajectory) is dropped — rounding a rotated integer
  grid can break a coincidence the reference relied on. Both arms' prompts
  carry the given coordinates, so only the tokens arm's *state blocks* are
  invariant. `prepare` freezes one file per condition under
  `ARTIFACT_ROOT/v1-ood-<ood hash>` (content fingerprint, not paths — the
  launcher prepares under /mnt, jobs read /workspace). Do not use
  `train_v1 --eval` for this: its `data_key` hashes the generator source,
  which changed after V1, so it would generate a *new* eval set.
  `run_v1_ood_jobs.py` prepares locally, then submits 2 arms x 2 seeds;
  `v1_ood --report` reads the H4 gate (`configs.H4_GATE`): paired
  retention on the long problems both arms solve at identity in every seed
  (tokens − table; raw drops would credit a joint collapse) on the far
  conditions translate-far and scale-up only — scale-down stays inside the
  training box and partly measures the table's 2-decimal print precision, so
  it is reported, not gated. `--smoke` uses 2 problems per bucket under
  `run/geostate/`. `rollout.evaluate` ends a torn last line before resuming
  (shared with `train_v1 --eval`).
- E0 request order is `cfg.models` order (every copy of one model is queued
  before the next), so `TEXT_MODELS` lists the gate-deciding models first.
