"""MLSpace job definitions for the vocabulary-factorization grids.

``configs.py`` stays the single source of truth for *what* each run is — arms,
inits, budgets, hyperparameters. This module adds only what a **job** needs on
top of that: which entrypoint runs it, how many GPUs, where its artifacts land,
and which file proves it finished. Adding an arm is still a one-line diff to
``configs.py``; it becomes launchable here automatically.

Two environment facts drive every path in this file, both learned from the
allocation rather than assumed:

**Worker containers mount the workspace at a different path.** This box sees
``/mnt/virtual_ai0001053-00054_SR004-nfs2/``; a job container sees
``/workspace-SR004.nfs2/``. Same bytes, different prefix — so every absolute
path handed to a job goes through ``to_job_path`` and every path the *launcher*
checks goes through ``to_local_path``. Mixing them up produces a job that dies
on a missing directory several minutes after submission.

**The conda env is on that same share.** ``/home/jovyan/.mlspace/envs`` is a
symlink onto it, so the interpreter is addressed by its workspace path and
invoked directly. Console scripts are avoided: their shebangs are baked to the
path the env was created at, which does not exist inside a worker.

**Data is not staged.** ``data/`` is gitignored and 10 GB, so the staged code
snapshot (tracked files at one commit) does not contain it. Jobs read the
corpus and token caches from the live repo by absolute path — they are
immutable inputs, so unlike code they cannot drift under a running job. What
they *can* do is race: ``prepare.get_tokens`` writes a cache on a miss, and two
jobs missing the same cache would write the same file at once. ``missing_inputs``
therefore refuses to launch anything whose caches are not already built.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from experiments.vocab_factorization.configs import (
    CALIBRATION_CANDIDATES,
    DISTILLS,
    HOST_ARM,
    SHRINK_CURRICULA,
    SHRINKS,
    TRANSPLANTS,
    TREE_TRANSPLANTS,
    TransplantTrainSpec,
    active_cache_path,
    fold_active_cache_path,
    fold_cache_path,
    token_cache_path,
)

# --- workspace path translation ------------------------------------------

WORKSPACE_LOCAL = "/mnt/virtual_ai0001053-00054_SR004-nfs2/"
WORKSPACE_JOB = "/workspace-SR004.nfs2/"
ENVS_LOCAL = "/home/jovyan/.mlspace/envs/"
ENVS_JOB = "/workspace-SR004.nfs2/d.tarasov/envs/"


def to_job_path(p: str | Path) -> str:
    """Rewrite a path on this box into the path a worker container sees."""
    s = str(p)
    if s.startswith(ENVS_LOCAL):
        return s.replace(ENVS_LOCAL, ENVS_JOB, 1)
    return s.replace(WORKSPACE_LOCAL, WORKSPACE_JOB, 1)


def to_local_path(p: str | Path) -> str:
    """The inverse: a container path as this box can stat it."""
    return str(p).replace(WORKSPACE_JOB, WORKSPACE_LOCAL, 1)


# --- out-of-workdir artifacts --------------------------------------------

ARTIFACTS_ROOT = "/workspace-SR004.nfs2/d.tarasov/gigaevo-artifacts"
STAGING_ROOT = f"{ARTIFACTS_ROOT}/code"
RESULTS_ROOT = f"{ARTIFACTS_ROOT}/results"
CKPT_ROOT = f"{ARTIFACTS_ROOT}/ckpt"

HF_HOME = "/workspace-SR004.nfs2/.cache/huggingface"
ENV_PREFIX = f"{ENVS_JOB}gigaevo-core/bin"
BASE_IMAGE = "cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.7.0:0.0.41"

# From `mls job instance_types` on this allocation (2026-08-28). Project-local
# by design: another allocation exposes different names and RAM tiers.
INSTANCE_TYPES_BY_NUM_GPUS = {n: f"a100.{n}gpu" for n in range(1, 9)}

# Every entrypoint in this study is a single process on a single card: the
# frozen 1.7B body plus one embedding table fits in 80 GB with room to spare,
# and nothing here is data-parallel. Parallelism is across *jobs*, not within.
NUM_GPUS = 1


@dataclass(frozen=True)
class JobSpec:
    """One MLSpace job: an entrypoint invocation plus its completion marker."""

    name: str
    module: str
    description: str
    args: tuple[str, ...] = ()
    arms: tuple[str, ...] = ()  # token caches this job reads
    # Arms whose *case-folded* caches this job reads. A separate field rather
    # than a flag on ``arms`` because the folded stream is a different encoding
    # of the same arm, with its own cache files and its own active set — and
    # both are read-through, so both have to be preflighted.
    fold_arms: tuple[str, ...] = ()
    num_gpus: int = NUM_GPUS
    marker: str = ""  # container path whose existence means "done"
    # Artifacts an *earlier wave* must have produced. Unlike ``arms``, a missing
    # one is not an error: it means the upstream job has not finished yet, which
    # is the normal state of a wave schedule. The launcher skips and says so.
    requires: tuple[str, ...] = ()
    priority_class: str | None = None

    def command(self, data_dir: str, results_dir: str) -> list[str]:
        return [
            f"{ENV_PREFIX}/python",
            "-m",
            self.module,
            "--corpus",
            f"{data_dir}/corpus",
            "--arms-dir",
            f"{data_dir}/arms",
            "--tokens-dir",
            f"{data_dir}/tokens",
            "--out",
            results_dir,
            *self.args,
        ]


# --- the registry ---------------------------------------------------------


def transplant_jobs(results_root: str = RESULTS_ROOT) -> list[JobSpec]:
    """One job per row of ``configs.TRANSPLANTS``.

    ``orig-frozen`` is eval-only (``train=False``) and so is the cheapest job in
    the study; the trained arms are ~1 GPU-hour each. All of them now write a
    checkpoint, which is what lets a later downstream evaluation ride along on a
    run scheduled for some other reason.
    """
    out = f"{results_root}/transplant"
    return [
        JobSpec(
            name=f"transplant-{t.name}",
            module="experiments.vocab_factorization.train_transplant",
            description=(
                f"transplant {t.name} ({t.arm}, {t.init} init"
                f"{', eval only' if not t.train else ''})"
            ),
            args=("--exp", t.name, "--ckpt-dir", CKPT_ROOT, "--reserve-gb", "24"),
            arms=(t.arm,),
            marker=f"{out}/{t.name}.json",
        )
        for t in TRANSPLANTS
    ]


def distill_jobs(results_root: str = RESULTS_ROOT) -> list[JobSpec]:
    """Stage 0's offline-distillation grid.

    Kept launchable because the grid is the reproduction path for a *published*
    negative result, not because anything downstream is waiting on it.
    """
    out = f"{results_root}/distill"
    return [
        JobSpec(
            name=f"distill-{d.name}",
            module="experiments.vocab_factorization.distill_embed",
            description=f"distill {d.name} ({d.method})",
            args=("--exp", d.name, "--reserve-gb", "24"),
            arms=(HOST_ARM,),
            marker=f"{out}/{d.name}.json",
        )
        for d in DISTILLS
    ]


def tree_jobs(results_root: str = RESULTS_ROOT) -> list[JobSpec]:
    """Stage 1: the feature-routed grid, one job per arm.

    The flat control these are measured against is ``transplant-bpe8k-t``, which
    the ``transplant`` group already launches under the active-vocab mask. No
    arm here budgets a second copy of it — that was the point of sequencing
    hardening around runs that were happening anyway.
    """
    out = f"{results_root}/transplant"
    return [
        JobSpec(
            name=f"tree-{t.name}",
            module="experiments.vocab_factorization.train_transplant",
            description=f"stage1 {t.name} ({t.arm}, x{t.budget_ratio} floats)",
            args=("--exp", t.name, "--ckpt-dir", CKPT_ROOT, "--reserve-gb", "24"),
            arms=(t.arm,),
            marker=f"{out}/{t.name}.json",
        )
        for t in TREE_TRANSPLANTS
    ]


def shrink_jobs(results_root: str = RESULTS_ROOT) -> list[JobSpec]:
    """Track 4: one job per rung of the progressive-shrink curriculum.

    ``shrink0-mask`` is eval-only (``train=False``) and costs what
    ``orig-frozen`` cost — 377 s on a worker — because it *is* ``orig-frozen``,
    reached through the shrink machinery instead of through the mask. It has to
    reproduce 0.7796 before any rung below it means anything, which is why it is
    a job rather than a unit test: the assertion is about the real model.

    The trained rungs ``requires`` that reference. Nothing reads the file — the
    number is in ``configs.MASKED_NATIVE_BPB`` — but a curve whose y-axis is
    "distance from the masked native reference" is uninterpretable before the
    reference exists, and refusing to spend 3 GPU-hours producing an
    uninterpretable curve is exactly what the mechanism is for.
    """
    out = f"{results_root}/transplant"
    return [
        JobSpec(
            name=f"shrink-{s.name}",
            module="experiments.vocab_factorization.train_transplant",
            description=(
                f"shrink {s.name} (stage {s.stage}"
                f"{', eval only' if not s.train else ''})"
            ),
            args=("--exp", s.name, "--ckpt-dir", CKPT_ROOT, "--reserve-gb", "24"),
            arms=(s.arm,),
            fold_arms=(
                (s.arm,)
                if any(
                    r.rule == "case_fold"
                    for r in SHRINK_CURRICULA[s.curriculum][: s.stage]
                )
                else ()
            ),
            marker=f"{out}/{s.name}.json",
            requires=(
                () if not s.train else (f"{RESULTS_ROOT}/transplant/orig-frozen.json",)
            ),
        )
        for s in SHRINKS
    ]


def calibration_jobs(results_root: str = RESULTS_ROOT) -> list[JobSpec]:
    """Track 3: one job per ladder candidate, then an analysis pass.

    Fanned out rather than looped in one job because the whole point of the
    track is to be cheap relative to the ~173 GPU-hour decision it gates, and 16
    ten-minute jobs finish in ten minutes on this allocation. The rho is
    computed afterwards by ``--analyze`` over whatever landed, so a candidate
    that fails does not take the verdict with it.
    """
    out = f"{results_root}/calibration"
    return [
        JobSpec(
            name=f"calib-{c.name}",
            module="experiments.vocab_factorization.calibrate",
            description=f"calibrate {c.name}"
            + (" [bad anchor]" if c.is_anchor else ""),
            args=("--candidate", c.name, "--reserve-gb", "24"),
            arms=("bpe8k",),
            marker=f"{out}/{c.name}.json",
        )
        for c in CALIBRATION_CANDIDATES
    ]


def hellaswag_jobs(results_root: str = RESULTS_ROOT) -> list[JobSpec]:
    """Transplant-hardening: the downstream task, one job per arm.

    ``orig-frozen`` is scored from the host table with ``--native`` because it
    writes no checkpoint (``train=False``), and it is the arm every other one is
    compared against. The trained arms are scored from the checkpoints their
    training jobs saved — which is the whole point of the hardening track
    budgeting no dedicated re-runs.
    """
    out = f"{results_root}/hellaswag"
    jobs = [
        JobSpec(
            name="hellaswag-orig-frozen",
            module="experiments.vocab_factorization.hellaswag",
            description="hellaswag orig-frozen (native host, no checkpoint)",
            args=("--native", "--reserve-gb", "24"),
            arms=(HOST_ARM,),
            marker=f"{out}/orig-frozen.json",
        )
    ]
    jobs += [
        JobSpec(
            name=f"hellaswag-{t.name}",
            module="experiments.vocab_factorization.hellaswag",
            description=f"hellaswag {t.name} ({t.arm})",
            args=("--ckpt", f"{CKPT_ROOT}/{t.name}.pt", "--reserve-gb", "24"),
            arms=(t.arm,),
            marker=f"{out}/{t.name}.json",
            requires=(f"{CKPT_ROOT}/{t.name}.pt",),
        )
        for t in TRANSPLANTS
        if t.train
    ]
    return jobs


def collect_jobs(
    filter_name: str | None = None,
    groups: tuple[str, ...] = ("transplant",),
    results_root: str = RESULTS_ROOT,
) -> list[JobSpec]:
    """The launchable set. ``distill`` is off by default — stage 0 is closed."""
    builders = {
        "transplant": transplant_jobs,
        "distill": distill_jobs,
        "hellaswag": hellaswag_jobs,
        "tree": tree_jobs,
        "shrink": shrink_jobs,
        "calibration": calibration_jobs,
    }
    jobs: list[JobSpec] = []
    for g in groups:
        if g not in builders:
            raise KeyError(f"unknown job group {g!r}, have {sorted(builders)}")
        jobs.extend(builders[g](results_root))
    if filter_name:
        jobs = [j for j in jobs if filter_name in j.name]
    return jobs


# --- input preflight ------------------------------------------------------


def missing_inputs(spec: JobSpec, data_dir_local: Path) -> list[str]:
    """Caches this job would have to *build*, and so must not build in parallel.

    ``get_tokens`` and ``active_vocab.load_or_build`` are read-through caches
    over a directory several jobs share. Building on the launcher, once, is
    seconds of CPU; discovering the race after a night of jobs is not.
    """
    tspec = TransplantTrainSpec()
    tokens = data_dir_local / "tokens"
    missing = []
    for arm in spec.arms:
        want = [
            token_cache_path(tokens, arm, "train", tspec.train_bytes),
            token_cache_path(tokens, arm, "valid", None),
            active_cache_path(tokens, arm),
        ]
        missing += [str(p) for p in want if not p.exists()]
    for arm in spec.fold_arms:
        want = [
            fold_cache_path(tokens, arm, "train", tspec.train_bytes),
            fold_active_cache_path(tokens, arm),
        ]
        missing += [str(p) for p in want if not p.exists()]
    return missing


def missing_requirements(spec: JobSpec) -> list[str]:
    """Upstream artifacts this job needs that do not exist yet."""
    return [r for r in spec.requires if not Path(to_local_path(r)).exists()]


def prewarm_command(arm: str, fold: bool = False) -> str:
    """What to run to fix what ``missing_inputs`` reports.

    The folded caches are built by ``shrink --curriculum fold``, which needs them
    and so writes them; it also prints the curve they produce, which is the
    thing worth reading before spending GPU time on it.
    """
    if fold:
        return (
            "PYTHONPATH=. python -m experiments.vocab_factorization.shrink "
            "--curriculum fold"
        )
    return (
        "PYTHONPATH=. python -m experiments.vocab_factorization.active_vocab "
        f"--arm {arm}"
    )


@dataclass
class Layout:
    """Where a launch reads from and writes to, in both path worlds."""

    repo_local: Path
    commit: str
    sanity: bool = False

    @property
    def data_job(self) -> str:
        return f"{to_job_path(self.repo_local)}/data"

    @property
    def data_local(self) -> Path:
        return self.repo_local / "data"

    @property
    def results_root(self) -> str:
        return f"{RESULTS_ROOT}_sanity" if self.sanity else RESULTS_ROOT

    @property
    def staged_job(self) -> str:
        return f"{STAGING_ROOT}/{self.commit}"

    @property
    def staged_local(self) -> Path:
        return Path(to_local_path(self.staged_job))
