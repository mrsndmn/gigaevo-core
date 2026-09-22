"""Stage and submit the frozen feedback evaluation to MLSpace.

--dry prints the planned manifest without writes or submission. The installed
mls CLI is used as the read-only API adapter so this launcher runs in the same
Python environment as the evaluator. No token is stored in the job manifest.
"""

from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import re
import shlex
import subprocess

from experiments.vocab_factorization.jobs import (
    BASE_IMAGE,
    ENV_PREFIX,
    STAGING_ROOT,
    to_job_path,
    to_local_path,
)
from experiments.vocab_factorization.run_train_jobs import stage_repo

from .configs import REPO_ROOT
from .credentials import resolve
from .feedback_eval import SPEC, comparison, prepare, write

MLS = "/home/jovyan/.mlspace/envs/arkhip/bin/mls"
JOB_ID = re.compile(
    r"lm-mpi-job-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)
DESCRIPTION = f"[gigaevo/noregeo_feedback] {SPEC.name}"


def git(*args):
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), *args], text=True
    ).strip()


def mls(*args):
    proc = subprocess.run(
        [MLS, "job", *args, "-O", "json"], capture_output=True, text=True, check=True
    )
    value = json.loads(proc.stdout)
    if value.get("error_code") or value.get("error_message"):
        raise RuntimeError(
            f"MLS request failed: {value.get('error_code')}: {value.get('error_message')}"
        )
    return value


def normalized(description):
    return " ".join(re.sub(r"[#@]\S+", "", description).split())


def payload(commit, out, credential_file):
    workdir = f"{STAGING_ROOT}/{commit}"
    command = [
        f"{ENV_PREFIX}/python",
        "-m",
        "experiments.noregeo_synth.feedback_eval",
        "--run",
        "--out",
        to_job_path(out),
    ]
    script = f"set -eu\ncd {shlex.quote(workdir)}\n" + shlex.join(command)
    return {
        "job": {
            "description": DESCRIPTION,
            "type": "binary_exp",
            "shm_size_class": "medium",
            "environment": {
                "image": BASE_IMAGE,
                "variables": {
                    "PYTHONPATH": workdir,
                    "PYTHONUNBUFFERED": "1",
                    "MPLBACKEND": "Agg",
                    "AIRI_ENV_FILE": to_job_path(credential_file),
                },
            },
            "resource": {
                "instance_type": SPEC.instance_type,
                "workers": 1,
                "processes": 1,
            },
            "script": script,
        }
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args(argv)
    commit = git("rev-parse", "HEAD")
    out = Path(SPEC.artifact_root) / SPEC.name
    creds = resolve()
    if creds.source == "environment":
        raise RuntimeError(
            "worker needs a persistent AIRI_ENV_FILE; environment-only credentials cannot be staged"
        )
    plan = payload(commit, out, creds.source)
    if args.dry:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        print(
            f"Frozen cohort: {SPEC.task_count} tasks; endpoint concurrency: {SPEC.workers}"
        )
        return 0
    if git("status", "--porcelain"):
        raise RuntimeError("commit changes before staging the evaluation")
    out.mkdir(parents=True, exist_ok=True)
    # Local/NFS advisory lock plus API-side in-progress lookup. Fail closed if
    # the API query is unavailable; never submit blindly.
    with (out / "submission.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare(out)
        if (out / "complete.json").exists():
            print(f"Already complete: {out}")
            return 0
        active = mls("list", "-s", "Pending", "-s", "Running", "-l", "6000")
        for job in active["jobs"]:
            if normalized(job.get("job_desc", "")) == DESCRIPTION:
                name = job["job_name"]
                if not JOB_ID.fullmatch(name):
                    raise RuntimeError("MLS returned an invalid full job ID")
                print(f"Already active: {name}")
                return 0
        stage_repo(REPO_ROOT, Path(to_local_path(f"{STAGING_ROOT}/{commit}")))
        # JSON is a YAML subset accepted by the mls manifest reader.
        manifest = out / "job-manifest.json"
        write(manifest, plan)
        comparison(out)
        result = mls("submit", "--config", str(manifest))
        name = result.get("job_name")
        if not isinstance(name, str) or not JOB_ID.fullmatch(name):
            write(out / "submission-response.json", result)
            raise RuntimeError(
                "submission did not return a complete MLS job ID; inspect response before retrying"
            )
        write(
            out / "submission.json",
            {"job_name": name, "commit": commit, "artifact_root": str(out)},
        )
        print(f"Submitted: {name}\nArtifacts: {out}\nHTML: {out / 'comparison.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
