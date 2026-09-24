"""Stage and submit the V1 arms to MLSpace: 2 arms x 2 seeds (AC-17).

    python -m experiments.geostate.run_v1_jobs --dry
    python -m experiments.geostate.run_v1_jobs
    python -m experiments.geostate.run_v1_jobs --arm tokens --seed 1

Same pattern as `run_e0_jobs`: commit-hash staging, external artifacts root,
dedup on active description, `HF_HOME` pointing at the shared cache that holds
Qwen3-1.7B. Each job prepares the shared data (once, under a file lock),
trains its arm and rolls out the evaluation set, writing `metrics.json`;
`v1_report.py` aggregates the four runs and reads the gate.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import hashlib
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

from .client import write_json
from .configs import ARTIFACT_ROOT, REPO_ROOT, V1, V1Config
from .run_e0_jobs import JOB_ID, git, mls, normalized


def protocol_hash_v1(cfg: V1Config) -> str:
    """Everything that shapes a run except arm and seed."""
    import inspect

    from . import features, sequence

    material = json.dumps(
        {
            "config": asdict(cfg),
            "sequence": inspect.getsource(sequence),
            "features": inspect.getsource(features),
        },
        sort_keys=True,
    )
    return hashlib.sha256(material.encode()).hexdigest()[:12]


def description_for(cfg: V1Config, arm: str, seed: int) -> str:
    return f"[gigaevo/geostate] {cfg.name} {arm} s{seed}"


def run_dir(cfg: V1Config, arm: str, seed: int) -> Path:
    return Path(ARTIFACT_ROOT) / "v1" / f"{arm}-s{seed}-{protocol_hash_v1(cfg)}"


def payload(commit: str, out: Path, cfg: V1Config, arm: str, seed: int) -> dict:
    workdir = f"{STAGING_ROOT}/{commit}"
    command = [
        f"{ENV_PREFIX}/python",
        "-m",
        "experiments.geostate.train_v1",
        "--arm",
        arm,
        "--seed",
        str(seed),
        "--out",
        to_job_path(out),
        "--data-root",
        to_job_path(Path(ARTIFACT_ROOT) / "v1"),
        "--train",
        "--eval",
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


def submit_one(cfg: V1Config, arm: str, seed: int, commit: str, dry: bool) -> int:
    out = run_dir(cfg, arm, seed)
    plan = payload(commit, out, cfg, arm, seed)
    if dry:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    out.mkdir(parents=True, exist_ok=True)
    description = description_for(cfg, arm, seed)
    with (out / "submission.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / "metrics.json").exists():
            print(f"Already complete: {out}")
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
        manifest = out / "job-manifest.json"
        write_json(manifest, plan)
        result = mls("submit", "--config", str(manifest))
        name = result.get("job_name")
        if not isinstance(name, str) or not JOB_ID.fullmatch(name):
            write_json(out / "submission-response.json", result)
            raise RuntimeError(
                "submission did not return a complete MLS job ID; inspect response "
                "before retrying"
            )
        write_json(
            out / "submission.json",
            {"job_name": name, "commit": commit, "artifact_root": str(out)},
        )
        print(f"Submitted: {name}\nArtifacts: {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=V1.arms, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args(argv)
    cfg = V1
    commit = git("rev-parse", "HEAD")
    if not args.dry and git("status", "--porcelain"):
        raise RuntimeError("commit changes before staging the training jobs")
    arms = [args.arm] if args.arm else list(cfg.arms)
    seeds = [args.seed] if args.seed is not None else list(cfg.seeds)
    for arm in arms:
        for seed in seeds:
            submit_one(cfg, arm, seed, commit, args.dry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
