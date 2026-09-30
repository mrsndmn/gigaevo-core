"""Stage and submit the H4 evaluation to MLSpace: 2 arms x 2 seeds, no training.

    python -m experiments.geostate.run_v1_ood_jobs --dry
    python -m experiments.geostate.run_v1_ood_jobs
    python -m experiments.geostate.run_v1_ood_jobs --arm tokens --seed 1

Same pattern as `run_v1_jobs`: commit-hash staging, external artifacts root,
dedup on active description (which carries the ood hash), `HF_HOME` pointing
at the shared cache. The pose conditions are frozen here, before anything is
submitted, so the four jobs only read them; each job loads its V1 adapters
and rolls out every condition (`v1_ood --eval`, resumable). Read with
`python -m experiments.geostate.v1_ood --report`.
"""

from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import shlex

from experiments.vocab_factorization.jobs import (
    BASE_IMAGE,
    ENV_PREFIX,
    HF_HOME,
    STAGING_ROOT,
    to_job_path,
    to_local_path,
)
from experiments.vocab_factorization.run_train_jobs import stage_repo

from .client import read_json, write_json
from .configs import REPO_ROOT, V1, V1_OOD, V1OODConfig
from .run_e0_jobs import JOB_ID, git, mls, normalized
from .v1_ood import (
    default_out,
    default_source_root,
    ood_hash,
    prepare,
    run_dir,
    source_dir,
)


def description_for(cfg: V1OODConfig, arm: str, seed: int) -> str:
    return f"[gigaevo/geostate] {cfg.name}/{ood_hash(cfg)} {arm} s{seed}"


def payload(
    commit: str, out: Path, source_root: Path, cfg: V1OODConfig, arm: str, seed: int
) -> dict:
    workdir = f"{STAGING_ROOT}/{commit}"
    command = [
        f"{ENV_PREFIX}/python",
        "-m",
        "experiments.geostate.v1_ood",
        "--eval",
        "--arm",
        arm,
        "--seed",
        str(seed),
        "--out",
        to_job_path(out),
        "--source-root",
        to_job_path(source_root),
    ]
    script = f"cd {shlex.quote(workdir)} && {shlex.join(command)}"
    return {
        "job": {
            "description": description_for(cfg, arm, seed),
            "type": "binary_exp",
            "shm_size_class": "medium",
            "environment": {
                "image": BASE_IMAGE,
                "variables": {
                    "PYTHONPATH": workdir,
                    "PYTHONUNBUFFERED": "1",
                    "MPLBACKEND": "Agg",
                    "HF_HOME": HF_HOME,
                    "HF_HUB_OFFLINE": "1",
                    "TOKENIZERS_PARALLELISM": "false",
                },
            },
            "resource": {
                "instance_type": cfg.instance_type,
                "workers": 1,
                "processes": 1,
            },
            "script": script,
        }
    }


def submit_one(
    cfg: V1OODConfig,
    arm: str,
    seed: int,
    commit: str,
    out: Path,
    source_root: Path,
    dry: bool,
) -> int:
    plan = payload(commit, out, source_root, cfg, arm, seed)
    if dry:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    adapters = source_dir(cfg, source_root, arm, seed) / "adapters.pt"
    if not adapters.exists():
        raise FileNotFoundError(f"no V1 adapters at {adapters}")
    run = run_dir(out, arm, seed)
    run.mkdir(parents=True, exist_ok=True)
    description = description_for(cfg, arm, seed)
    with (run / "submission.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        done = run / "run.json"
        if done.exists() and read_json(done).get("complete"):
            print(f"Already complete: {run}")
            return 0
        active = mls("list", "-s", "Pending", "-s", "Running", "-l", "6000")
        for job in active["jobs"]:
            if normalized(job.get("job_desc", "")) == description:
                name = job["job_name"]
                if not JOB_ID.fullmatch(name):
                    raise RuntimeError("MLS returned an invalid full job ID")
                print(f"Already active: {name}")
                return 0
        stage_repo(REPO_ROOT, Path(to_local_path(f"{STAGING_ROOT}/{commit}")))
        manifest = run / "job-manifest.json"
        write_json(manifest, plan)
        result = mls("submit", "--config", str(manifest))
        name = result.get("job_name")
        if not isinstance(name, str) or not JOB_ID.fullmatch(name):
            write_json(run / "submission-response.json", result)
            raise RuntimeError(
                "submission did not return a complete MLS job ID; inspect response "
                "before retrying"
            )
        write_json(
            run / "submission.json",
            {"job_name": name, "commit": commit, "artifact_root": str(out)},
        )
        print(f"Submitted: {name}\nArtifacts: {run}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=V1.arms, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args(argv)
    cfg = V1_OOD
    commit = git("rev-parse", "HEAD")
    if not args.dry and git("status", "--porcelain"):
        raise RuntimeError("commit changes before staging the evaluation jobs")
    out = default_out(cfg)
    source_root = default_source_root()
    if not args.dry:
        prepare(cfg, out, source_root)  # frozen before any job reads it
    arms = [args.arm] if args.arm else list(V1.arms)
    seeds = [args.seed] if args.seed is not None else list(V1.seeds)
    for arm in arms:
        for seed in seeds:
            submit_one(cfg, arm, seed, commit, out, source_root, args.dry)
    if args.dry:
        print(f"Would freeze pose conditions under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
