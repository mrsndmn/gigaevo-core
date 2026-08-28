"""Submit checkpoint-scoped evaluation jobs to MLSpace.

Evaluation here is keyed on a **checkpoint**, not on an experiment name: the
launcher enumerates what is actually on disk under the artifacts root and
submits one job per saved model. That ordering is deliberate. It is what lets a
question asked next week attach itself to a run that happened this week — the
capability study 2 did not have, and the reason ``transplant-hardening`` can
budget zero dedicated re-runs.

Until a training job has written a checkpoint there is nothing to evaluate and
this exits 0 with "nothing to evaluate" — a fresh artifacts root is a normal
state, not a failure.

Usage::

    python -m experiments.vocab_factorization.run_eval --dry
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import sys

from experiments.vocab_factorization.jobs import (
    BASE_IMAGE,
    CKPT_ROOT,
    ENV_PREFIX,
    HF_HOME,
    INSTANCE_TYPES_BY_NUM_GPUS,
    NUM_GPUS,
    Layout,
    to_job_path,
    to_local_path,
)
from experiments.vocab_factorization.run_train_jobs import (
    _dirty_tracked,
    _git,
    normalize_job_desc,
    redact_payload,
    stage_repo,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--profile", default="default")
    p.add_argument("--base_image", default=BASE_IMAGE)
    p.add_argument("--author_name", default="ARKHIP (d.tarasov)")
    p.add_argument("--telegram_nick", default="mrsndmn")
    p.add_argument("--checkpoints_dir", default=CKPT_ROOT)
    p.add_argument("--dry", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--sanity", action="store_true")
    p.add_argument("--filter", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    commit = _git(repo_root, "rev-parse", "HEAD")
    layout = Layout(repo_local=repo_root, commit=commit, sanity=args.sanity)

    ckpt_local = Path(to_local_path(args.checkpoints_dir))
    ckpts = sorted(ckpt_local.glob("*.pt")) if ckpt_local.is_dir() else []
    if args.filter:
        ckpts = [c for c in ckpts if args.filter in c.stem]
    if not ckpts:
        print(f"Nothing to evaluate: no *.pt under {ckpt_local}")
        return 0

    if args.dry:
        workdir = to_job_path(repo_root)
    else:
        dirty = _dirty_tracked(repo_root)
        if dirty:
            print(
                f"\033[31mWorking tree is dirty (HEAD={commit[:8]}). Commit or "
                "stash first.\033[0m",
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

    results_dir = f"{layout.results_root}/transplant"
    launched: list[dict] = []
    planned = 0

    for ckpt in ckpts:
        planned += 1
        name = ckpt.stem
        marker_local = Path(to_local_path(f"{results_dir}/{name}.eval.json"))
        if marker_local.exists() and not args.force:
            print(f"\033[33mSkip (eval exists):\033[0m {marker_local}")
            continue

        cmd = [
            f"{ENV_PREFIX}/python",
            "-m",
            "experiments.vocab_factorization.eval_transplant",
            "--ckpt",
            f"{args.checkpoints_dir}/{ckpt.name}",
            "--corpus",
            f"{layout.data_job}/corpus",
            "--arms-dir",
            f"{layout.data_job}/arms",
            "--tokens-dir",
            f"{layout.data_job}/tokens",
            "--out",
            results_dir,
            "--reserve-gb",
            "24",
        ]
        script = (
            f"cd {shlex.quote(workdir)} && "
            f"export PYTHONPATH={shlex.quote(workdir)} && {shlex.join(cmd)}"
        )

        job_desc = (
            f"[gigaevo/vocab_factorization]: eval checkpoint {name} {commit[:8]} "
            f"#{args.author_name} #rnd #notify_completed @{args.telegram_nick}"
        )
        if normalize_job_desc(job_desc) in in_progress:
            print(f"\033[33mSkip (already queued):\033[0m {job_desc}")
            continue

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
            "instance_type": INSTANCE_TYPES_BY_NUM_GPUS[NUM_GPUS],
            "region": extra["region"],
            "type": "binary_exp",
            "shm_size_class": "medium",
            "base_image": args.base_image,
            "n_workers": 1,
            "processes_per_worker": 1,
        }

        print(f"\033[32mWould launch [eval]:\033[0m {job_desc}")
        print(f"\033[90m  {script}\033[0m")
        if args.dry:
            print(json.dumps(redact_payload(payload), indent=2))
            continue

        result = run_job_with_retry(client, payload, profile=args.profile)
        job_name = result.get("job_name") if isinstance(result, dict) else None
        if job_name:
            launched.append({"job_name": job_name, "name": name})
        print("result", result)

    print(f"\nPlanned: {planned} | Launched: {len(launched)}")
    print("__EVAL_JOBS_JSON__")
    print(json.dumps({"jobs": launched, "launched": len(launched)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
