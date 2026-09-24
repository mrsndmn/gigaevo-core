"""Stage and submit one E0 consistency config to MLSpace.

    python -m experiments.geostate.run_e0_jobs --config e0-synthetic-text --dry
    python -m experiments.geostate.run_e0_jobs --config e0-synthetic-text
    python -m experiments.geostate.run_e0_jobs --config e0-synthetic-text --limit 12

Modelled on `experiments.noregeo_synth.run_recovery_eval`: the committed tree
is staged by commit hash (a live edit cannot perturb a running job), outputs
go to the external artifacts root, the worker gets the credentials env file
and the TLS bundle (`SSL_CERT_FILE`), and an active job with the same
description is never submitted twice. `--dry` prints the manifest and submits
nothing. The installed `mls` CLI (arkhip env) is the API adapter, so this
launcher itself runs in the evaluator's environment.

The job runs `consistency --run` and then `--summary`; the summary can also be
rebuilt locally at any time from the NFS outputs.
"""

from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import re
import shlex
import subprocess

from experiments.noregeo_synth.credentials import resolve
from experiments.vocab_factorization.jobs import (
    BASE_IMAGE,
    ENV_PREFIX,
    STAGING_ROOT,
    to_job_path,
    to_local_path,
)
from experiments.vocab_factorization.run_train_jobs import stage_repo

from .client import ca_bundle, write_json
from .configs import (
    ARTIFACT_ROOT,
    DEFAULT_CONFIG,
    REPO_ROOT,
    TLS_BUNDLE_NAME,
    E0Config,
    config_named,
    experiment_configs,
)
from .consistency import prepare

MLS = "/home/jovyan/.mlspace/envs/arkhip/bin/mls"
JOB_ID = re.compile(
    r"lm-mpi-job-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), *args], text=True
    ).strip()


def parse_mls_output(output: str) -> dict:
    # The installed CLI prints a Python-dict diagnostic before its JSON
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


def mls(*args: str) -> dict:
    proc = subprocess.run(
        [MLS, "job", *args, "-O", "json"], capture_output=True, text=True, check=True
    )
    value = parse_mls_output(proc.stdout)
    if value.get("error_code") or value.get("error_message"):
        raise RuntimeError(
            f"MLS request failed: {value.get('error_code')}: {value.get('error_message')}"
        )
    return value


def normalized(description: str) -> str:
    return " ".join(re.sub(r"[#@]\S+", "", description).split())


def description_for(cfg: E0Config, limit: int | None) -> str:
    suffix = f" limit{limit}" if limit is not None else ""
    return f"[gigaevo/geostate] {cfg.name}{suffix}"


def artifact_dir(cfg: E0Config, limit: int | None) -> Path:
    name = cfg.name if limit is None else f"{cfg.name}-limit{limit}"
    return Path(ARTIFACT_ROOT) / name


def payload(
    commit: str,
    out: Path,
    credential_file: str,
    cfg: E0Config,
    limit: int | None = None,
) -> dict:
    workdir = f"{STAGING_ROOT}/{commit}"
    base = [f"{ENV_PREFIX}/python", "-m", "experiments.geostate.consistency"]
    common = ["--config", cfg.name, "--out", to_job_path(out)]
    run = [*base, *common, "--run", "--workers", str(cfg.workers)]
    if limit is not None:
        run += ["--limit", str(limit)]
    summary = [*base, *common, "--summary"]
    script = (
        f"cd {shlex.quote(workdir)} && {shlex.join(run)}; "
        f"status=$?; {shlex.join(summary)}; exit $status"
    )
    return {
        "job": {
            "description": description_for(cfg, limit),
            "type": "binary_exp",
            "shm_size_class": "medium",
            "environment": {
                "image": BASE_IMAGE,
                "variables": {
                    "PYTHONPATH": workdir,
                    "PYTHONUNBUFFERED": "1",
                    "MPLBACKEND": "Agg",
                    "AIRI_ENV_FILE": to_job_path(credential_file),
                    "SSL_CERT_FILE": to_job_path(out / TLS_BUNDLE_NAME),
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=", ".join(c.name for c in experiment_configs),
    )
    parser.add_argument("--limit", type=int, default=None, help="pilot: N problems")
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args(argv)
    cfg = config_named(args.config)
    commit = git("rev-parse", "HEAD")
    out = artifact_dir(cfg, args.limit)
    creds = resolve()
    if creds.source == "environment":
        raise RuntimeError(
            "worker needs a persistent AIRI_ENV_FILE; environment-only credentials "
            "cannot be staged"
        )
    plan = payload(commit, out, creds.source, cfg, args.limit)
    if args.dry:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        print(
            f"Would freeze inputs under {out}; endpoint concurrency {cfg.workers}; "
            f"models {', '.join(cfg.models)}"
        )
        return 0
    if git("status", "--porcelain"):
        raise RuntimeError("commit changes before staging the evaluation")
    out.mkdir(parents=True, exist_ok=True)
    description = description_for(cfg, args.limit)
    with (out / "submission.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare(cfg, out)  # frozen inputs, fingerprinted
        ca_bundle(out)
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


if __name__ == "__main__":
    raise SystemExit(main())
