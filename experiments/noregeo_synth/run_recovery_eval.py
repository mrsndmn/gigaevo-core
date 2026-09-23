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
from .feedback_eval import write
from .recovery_eval import SPEC, prepare, report

MLS = "/home/jovyan/.mlspace/envs/arkhip/bin/mls"
JOB_ID = re.compile(
    r"lm-mpi-job-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)
DESCRIPTION = f"[gigaevo/noregeo_recovery] {SPEC.name}"


def git(*args):
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), *args], text=True
    ).strip()


def parse_mls_output(output):
    # This installed CLI prints a Python-dict diagnostic before its JSON
    # submission response. Parse the trailing complete JSON object only.
    output = re.sub(r"\x1b\[[0-9;]*m", "", output)
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", output):
        try:
            value, end = decoder.raw_decode(output[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and not output[match.start() + end :].strip():
            return value
    raise ValueError(
        "MLS CLI did not return a final JSON response; inspect job list before retrying"
    )


def mls(*args):
    proc = subprocess.run(
        [MLS, "job", *args, "-O", "json"], capture_output=True, text=True, check=True
    )
    value = parse_mls_output(proc.stdout)
    if value.get("error_code") or value.get("error_message"):
        raise RuntimeError(
            f"MLS request failed: {value.get('error_code')}: {value.get('error_message')}"
        )
    return value


def normalized(description):
    return " ".join(re.sub(r"[#@]\S+", "", description).split())


def payload(
    commit,
    out,
    credential_file,
    spec=SPEC,
    module="experiments.noregeo_synth.recovery_eval",
):
    description = f"[gigaevo/noregeo_recovery] {spec.name}"
    workdir = f"{STAGING_ROOT}/{commit}"
    command = [
        f"{ENV_PREFIX}/python",
        "-m",
        module,
        "--run",
        "--out",
        to_job_path(out),
    ]
    script = f"cd {shlex.quote(workdir)} && " + shlex.join(command)
    tls_env = {}
    if module == "experiments.noregeo_synth.recovery_extra":
        tls_env["SSL_CERT_FILE"] = to_job_path(Path(out) / "tls-ca-bundle.pem")
    return {
        "job": {
            "description": description,
            "type": "binary_exp",
            "shm_size_class": "medium",
            "environment": {
                "image": BASE_IMAGE,
                "variables": {
                    "PYTHONPATH": workdir,
                    "PYTHONUNBUFFERED": "1",
                    "MPLBACKEND": "Agg",
                    "AIRI_ENV_FILE": to_job_path(credential_file),
                    **tls_env,
                },
            },
            "resource": {
                "instance_type": spec.instance_type,
                "workers": 1,
                "processes": 1,
            },
            "script": script,
        }
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry", action="store_true")
    parser.add_argument(
        "--experiment", choices=("original", "extra15"), default="original"
    )
    args = parser.parse_args(argv)
    spec, prepare_run, report_run = SPEC, prepare, report
    module = "experiments.noregeo_synth.recovery_eval"
    if args.experiment == "extra15":
        from . import recovery_extra

        spec, prepare_run, report_run = (
            recovery_extra.SPEC,
            recovery_extra.prepare,
            recovery_extra.report,
        )
        module = "experiments.noregeo_synth.recovery_extra"
    description = f"[gigaevo/noregeo_recovery] {spec.name}"
    commit = git("rev-parse", "HEAD")
    out = Path(spec.artifact_root) / spec.name
    creds = resolve()
    if creds.source == "environment":
        raise RuntimeError(
            "worker needs a persistent AIRI_ENV_FILE; environment-only credentials cannot be staged"
        )
    plan = payload(commit, out, creds.source, spec, module)
    if args.dry:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        print(
            f"Frozen cohort: {getattr(spec, 'count', 75)} tasks; endpoint concurrency: {spec.workers}"
        )
        return 0
    if git("status", "--porcelain"):
        raise RuntimeError("commit changes before staging the evaluation")
    out.mkdir(parents=True, exist_ok=True)
    # Local/NFS advisory lock plus API-side in-progress lookup. Fail closed if
    # the API query is unavailable; never submit blindly.
    with (out / "submission.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare_run(out)
        if (out / "complete.json").exists() or (out / "stopped.json").exists():
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
        # JSON is a YAML subset accepted by the mls manifest reader.
        manifest = out / "job-manifest.json"
        write(manifest, plan)
        report_run(out)
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
        print(f"Submitted: {name}\nArtifacts: {out}\nHTML: {out / 'analysis.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
