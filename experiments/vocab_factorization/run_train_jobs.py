"""Submit the vocabulary-factorization grid to MLSpace — one job per arm.

The study's runs are single-GPU and mutually independent, so the schedule is
pure fan-out: N arms become N jobs on ``a100.1gpu``. This is what makes the
plan's wave structure real rather than aspirational — W2's stage-1 grid,
calibration and shrink curriculum are meant to run *at the same time*, and on
one contended local card they cannot.

Safety properties, in the order they bite:

* **dirty tree refused** — staging materializes from ``.git``, so uncommitted
  work would be silently dropped from what actually runs;
* **code staged per commit** — the job runs a clean checkout at
  ``<STAGING_ROOT>/<commit>``, so editing the working tree cannot perturb a job
  that has been queued for an hour;
* **inputs pre-warmed** — a job whose token caches are missing is refused rather
  than allowed to race a sibling building the same file;
* **idempotent** — an arm whose result JSON exists is skipped unless ``--force``;
* **dedup'd** — an arm already Pending/Running is skipped, matched on the job
  description with ``#tags``/``@nicks`` stripped.

Usage::

    python -m experiments.vocab_factorization.run_train_jobs --dry
    python -m experiments.vocab_factorization.run_train_jobs --filter orig-frozen
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from experiments.vocab_factorization.jobs import (
    BASE_IMAGE,
    ENV_PREFIX,
    HF_HOME,
    INSTANCE_TYPES_BY_NUM_GPUS,
    Layout,
    collect_jobs,
    missing_inputs,
    prewarm_command,
    to_job_path,
    to_local_path,
)

try:  # the shared mls library ships these; older installs may not
    from mls.manager.job.dedup import normalize_job_desc
except ImportError:  # pragma: no cover
    import re

    _TAGS = re.compile(r"[#@]\S+")

    def normalize_job_desc(desc: str) -> str:
        return " ".join(_TAGS.sub("", desc).split())


try:
    from mls.manager.job.redact import redact_payload
except ImportError:  # pragma: no cover

    def redact_payload(payload: dict, *, hints=("TOKEN", "SECRET", "KEY", "PASSWORD")):
        env = dict(payload.get("env_variables", {}))
        for k, v in env.items():
            if v and any(h in k.upper() for h in hints):
                env[k] = "<redacted>"
        return {**payload, "env_variables": env}


def _git(repo_root: str | Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _dirty_tracked(repo_root: str | Path) -> list[str]:
    return [
        ln
        for ln in _git(repo_root, "status", "--porcelain").splitlines()
        if ln and not ln.startswith("??")
    ]


def stage_repo(repo_root: Path, target: Path) -> None:
    """Copy ``.git`` + ``git reset --hard HEAD`` into target. Idempotent, atomic."""
    if target.is_dir():
        print(f"Reusing staged repo at: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.tmp.{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    print(f"Staging {repo_root} -> {target}")
    subprocess.run(["cp", "-a", str(repo_root / ".git"), str(tmp)], check=True)
    subprocess.run(["git", "-C", str(tmp), "reset", "--hard", "HEAD"], check=True)
    try:
        tmp.rename(target)
    except OSError:
        shutil.rmtree(tmp, ignore_errors=True)
        if not target.is_dir():
            raise


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--profile", default="default")
    p.add_argument("--base_image", default=BASE_IMAGE)
    p.add_argument("--author_name", default="ARKHIP (d.tarasov)")
    p.add_argument("--telegram_nick", default="mrsndmn")
    p.add_argument("--dry", action="store_true", help="print payloads, launch nothing")
    p.add_argument("--force", action="store_true", help="launch even if results exist")
    p.add_argument("--sanity", action="store_true", help="write to the sanity root")
    p.add_argument("--filter", default=None, help="substring filter on job name")
    p.add_argument(
        "--groups",
        nargs="*",
        default=["transplant"],
        help="job groups to launch (transplant, distill)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    commit = _git(repo_root, "rev-parse", "HEAD")
    layout = Layout(repo_local=repo_root, commit=commit, sanity=args.sanity)

    jobs = collect_jobs(args.filter, tuple(args.groups), layout.results_root)
    if not jobs:
        print("No jobs matched.", file=sys.stderr)
        return 1

    # Preflight the shared read-through caches before anything is submitted, so
    # a missing one is a message here rather than a write race out there.
    need = {m for j in jobs for m in missing_inputs(j, layout.data_local)}
    if need:
        arms = sorted({a for j in jobs for a in j.arms})
        print(
            "\033[31mMissing cached inputs — build them first:\033[0m", file=sys.stderr
        )
        for m in sorted(need):
            print(f"  {m}", file=sys.stderr)
        for a in arms:
            print(f"  $ {prewarm_command(a)}", file=sys.stderr)
        return 1

    if args.dry:
        # No staging in dry mode: the printed script points at the live repo, so
        # what you read is what a job would run from this commit.
        workdir = to_job_path(repo_root)
    else:
        dirty = _dirty_tracked(repo_root)
        if dirty:
            print(
                f"\033[31mWorking tree is dirty (HEAD={commit[:8]}). Commit or "
                "stash first — staging materializes from .git, so uncommitted "
                "changes would be silently dropped.\033[0m",
                file=sys.stderr,
            )
            print("\n".join(dirty), file=sys.stderr)
            return 1
        stage_repo(repo_root, layout.staged_local)
        workdir = layout.staged_job

    from mls.manager.job.utils import (
        get_in_progress_jobs,
        run_job_with_retry,
        training_job_api_from_profile,
    )

    client, extra = training_job_api_from_profile(args.profile)
    in_progress = {
        normalize_job_desc(j.get("job_desc", "")) for j in get_in_progress_jobs()
    }

    launched: list[dict] = []
    planned = 0

    for spec in jobs:
        planned += 1
        marker_local = Path(to_local_path(spec.marker))
        if marker_local.exists() and not args.force:
            print(f"\033[33mSkip (result exists):\033[0m {marker_local}")
            continue

        results_dir = str(Path(spec.marker).parent)
        cmd = spec.command(layout.data_job, results_dir)
        script = f"cd {shlex.quote(workdir)} && export PYTHONPATH={shlex.quote(workdir)} && {shlex.join(cmd)}"

        job_desc = (
            f"[gigaevo/vocab_factorization]: {spec.description} {commit[:8]} "
            f"#{args.author_name} #rnd #notify_completed @{args.telegram_nick}"
        )
        if normalize_job_desc(job_desc) in in_progress:
            print(f"\033[33mSkip (already queued):\033[0m {job_desc}")
            continue

        if spec.num_gpus not in INSTANCE_TYPES_BY_NUM_GPUS:
            print(f"\033[31mNo instance type for num_gpus={spec.num_gpus}\033[0m")
            return 1

        payload = {
            "script": script,
            "job_desc": job_desc,
            "env_variables": {
                "ENV_PREFIX": ENV_PREFIX,
                "WORKDIR": workdir,
                "HF_HOME": HF_HOME,
                "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
                "PYTHONUNBUFFERED": "1",
            },
            "instance_type": INSTANCE_TYPES_BY_NUM_GPUS[spec.num_gpus],
            "region": extra["region"],
            "type": "binary_exp",
            "shm_size_class": "medium",
            "base_image": args.base_image,
            "n_workers": 1,
            "processes_per_worker": 1,
            **({"priority_class": spec.priority_class} if spec.priority_class else {}),
        }

        print(f"\033[32mWould launch [{spec.num_gpus} GPU]:\033[0m {job_desc}")
        print(f"\033[90m  {script}\033[0m")
        if args.dry:
            print(json.dumps(redact_payload(payload), indent=2))
            continue

        result = run_job_with_retry(client, payload, profile=args.profile)
        job_name = result.get("job_name") if isinstance(result, dict) else None
        if job_name:
            launched.append(
                {"job_name": job_name, "name": spec.name, "marker": spec.marker}
            )
        print("result", result)

    print(f"\nPlanned: {planned} | Launched: {len(launched)}")
    print("__TRAINING_JOBS_JSON__")
    print(json.dumps({"jobs": launched, "launched": len(launched)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
