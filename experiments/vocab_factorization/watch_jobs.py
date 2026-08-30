"""Watch a batch of submitted MLSpace jobs and report when the wave is done.

``run_train_jobs`` returns as soon as the API accepts the payloads, which is the
right behaviour — the jobs outlive the launcher and the session — but it leaves
nobody watching. This polls until every job in a batch reaches a terminal state,
prints each transition as it happens, and ends with the numbers rather than just
a status: for a wave whose whole purpose is a BPB comparison, "5 completed" is
not the answer to any question anyone asked.

It is deliberately a *reader*. It never resubmits, never kills, and never
touches the artifacts — a watcher that repaired things would make "the wave
finished" mean something different every time it ran.

Run under the arkhip env, which is the one with ``mls``::

    PYTHONPATH=. /home/jovyan/.mlspace/envs/arkhip/bin/python \\
        -m experiments.vocab_factorization.watch_jobs --jobs-file run/jobs.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from experiments.vocab_factorization.jobs import to_local_path

POLL_DEFAULT = 120


def _status_map(profile: str) -> dict[str, str]:
    """job_name -> status for everything currently Pending or Running.

    Absence is the signal: ``get_in_progress_jobs`` is the only listing call
    that is cheap enough to poll, so a job that has fallen out of it has
    finished, one way or another.
    """
    from mls.manager.job.utils import get_in_progress_jobs

    return {
        j.get("job_name", ""): j.get("status", "?")
        for j in get_in_progress_jobs(client_profile=profile)
    }


def _final_status(job_name: str, profile: str) -> str:
    """The terminal status of one job, or '?' if the API will not say."""
    from mls.manager.job.utils import training_job_api_from_profile

    try:
        client, _ = training_job_api_from_profile(profile)
        info = client.get_job_status(job_name)
        if isinstance(info, dict):
            return str(info.get("status", "?"))
    except Exception as exc:  # the wave's outcome must not hinge on this call
        return f"?({type(exc).__name__})"
    return "?"


def summarize(jobs: list[dict]) -> list[str]:
    """Read each finished job's marker and pull out what the wave was for."""
    lines = []
    for j in jobs:
        marker = Path(to_local_path(j["marker"]))
        if not marker.exists():
            lines.append(f"  {j['name']:32s} NO RESULT at {marker}")
            continue
        d = json.loads(marker.read_text())
        # The wave is heterogeneous: transplant runs, HellaSwag scores and
        # calibration rungs land in the same batch and share no fields. Report
        # whichever this is rather than assuming, because a summary that raises
        # loses the whole wave's numbers over one job of the wrong kind.
        if "acc_norm" in d:
            # A shrink arm has no mask to be outside of — its inactive rows are
            # deleted — so it reports the codec's own coverage under a different
            # name. Same column, and the summary still does not raise.
            cov = d["active_vocab_coverage"]
            oov = cov.get("fraction_outside_active", cov.get("fraction_outside_codec"))
            lines.append(
                f"  {j['name']:32s} acc={d['acc']:.4f} acc_norm={d['acc_norm']:.4f} "
                f"oov={100 * oov:.2f}%"
            )
            continue
        if "L2_short_train_bpb" in d:
            lines.append(
                f"  {j['name']:32s} L0={d['L0_explained_variance']:.4f} "
                f"L1={d['L1_step0_bpb']:.4f} L2={d['L2_short_train_bpb']:.4f}"
            )
            continue
        clean = (d.get("eval.clean") or {}).get("bpb")
        if clean is None:
            lines.append(f"  {j['name']:32s} (no clean BPB in {marker.name})")
            continue
        step0 = d.get("step0_clean_bpb")
        tree = d.get("tree") or {}
        extra = (
            f" x{tree['compression']:.2f} ({tree['float_params'] / 1e6:.2f}M floats)"
            if tree
            else ""
        )
        mins = (d.get("train_seconds") or 0) / 60
        lines.append(
            f"  {j['name']:32s} clean={clean:.4f} step0={step0:.4f} "
            if step0 is not None
            else f"  {j['name']:32s} clean={clean:.4f} "
        )
        lines[-1] += f"steps={d.get('steps')} {mins:.0f}min{extra}"
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs-file", type=Path, required=True)
    ap.add_argument("--profile", default="default")
    ap.add_argument("--interval", type=int, default=POLL_DEFAULT)
    ap.add_argument("--timeout-hours", type=float, default=12.0)
    args = ap.parse_args()

    payload = json.loads(args.jobs_file.read_text())
    jobs = payload["jobs"] if isinstance(payload, dict) else payload
    pending = {j["job_name"]: j for j in jobs}
    print(f"watching {len(pending)} job(s), polling every {args.interval}s", flush=True)
    for j in jobs:
        print(f"  {j['name']:32s} {j['job_name']}", flush=True)

    seen: dict[str, str] = {}
    done: dict[str, str] = {}
    deadline = time.monotonic() + args.timeout_hours * 3600
    while pending and time.monotonic() < deadline:
        live = _status_map(args.profile)
        for name in list(pending):
            if name in live:
                if live[name] != seen.get(name):
                    seen[name] = live[name]
                    print(
                        f"[{time.strftime('%H:%M:%S')}] {pending[name]['name']} "
                        f"-> {live[name]}",
                        flush=True,
                    )
                continue
            # Gone from the in-progress listing: terminal.
            status = _final_status(name, args.profile)
            done[name] = status
            print(
                f"[{time.strftime('%H:%M:%S')}] {pending[name]['name']} "
                f"FINISHED ({status}) — {len(done)}/{len(jobs)}",
                flush=True,
            )
            pending.pop(name)
        if pending:
            time.sleep(args.interval)

    if pending:
        print(f"\nTIMED OUT with {len(pending)} still running:", flush=True)
        for name, j in pending.items():
            print(f"  {j['name']:32s} {name}", flush=True)

    ok = [n for n, s in done.items() if s == "completed"]
    print(
        f"\nWAVE DONE: {len(ok)}/{len(jobs)} completed, "
        f"{len(done) - len(ok)} other, {len(pending)} unresolved",
        flush=True,
    )
    for name, status in done.items():
        if status != "completed":
            print(
                f"  NOT COMPLETED: {pending.get(name, {}).get('name', name)} {status}"
            )
    print("\nResults:", flush=True)
    for line in summarize(jobs):
        print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
