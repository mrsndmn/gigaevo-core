"""One-shot local queue watcher using Arkhip's Telegram transport and routing.

Run in detached tmux with TELEGRAM_BOT_TOKEN inherited from Arkhip. This is not
an MLS job_wait registration. Pin --run-start to the restarted queue's timestamp.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import time

from .client import write_json
from .configs import ARTIFACT_ROOT


def terminal_status(state: dict, run_start: float, locked: bool) -> str | None:
    if state.get("started") != run_start:
        return "superseded"
    if state.get("status") in {"complete", "finished_with_errors"}:
        return state["status"]
    return None if locked else "interrupted"


def queue_locked(root: Path) -> bool:
    with (root / ".figure-queue.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False


def observe(root: Path, run_start: float) -> tuple[dict, str | None]:
    # Read after obtaining the controller lock (if free), keeping it until the
    # snapshot is classified. Otherwise a normal finish can look interrupted.
    with (root / ".figure-queue.lock").open("a+") as handle:
        locked = False
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            locked = True
        state = json.loads((root / "figure-queue-state.json").read_text())
        return state, terminal_status(state, run_start, locked)


def notification(state: dict, status: str) -> str:
    titles = {
        "complete": "GeoState: очередь картиночных задач завершена.",
        "finished_with_errors": "GeoState: очередь закончилась с ошибками; часть запросов требует повторного запуска.",
        "interrupted": "GeoState: процесс очереди остановился до завершения.",
        "superseded": "GeoState: наблюдаемая очередь заменена новым запуском; её наблюдатель остановлен.",
    }
    lines = [titles[status]]
    if status != "superseded":
        for name, entry in state.get("cohorts", {}).items():
            lines.append(f"{name}: {entry.get('status', 'unknown')}, сохранено {entry.get('evaluated', '—')} оценок")
    return "\n".join(lines)


def send_notification(registry: Path, project: str, message: str) -> None:
    import agenttg
    import agenttg.api
    import yaml

    token = os.environ["TELEGRAM_BOT_TOKEN"]  # Never use the obsolete YAML token.
    route = yaml.safe_load(registry.read_text())["projects"][project]
    # agenttg logs raw HTTP exceptions, which may contain the token in the URL.
    # This watcher is single-threaded; suppress transport logs only during send.
    logger = agenttg.api.logger
    was_disabled = logger.disabled
    logger.disabled = True
    try:
        responses = agenttg.send_reply(
            token=token, chat_id=str(route["telegram_chat_id"]), text=message,
            thread_id=route.get("telegram_thread_id", 0),
        )
    finally:
        logger.disabled = was_disabled
    if not responses or not all(response.json().get("ok") for response in responses):
        raise RuntimeError("Telegram did not acknowledge notification")


def watch(root: Path, run_start: float, registry: Path, project: str,
          interval: float = 30) -> None:
    if interval <= 0:
        raise ValueError("interval must be positive")
    marker = root / f"figure-queue-watch-{run_start:.6f}.json"
    with marker.with_suffix(".lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if marker.exists():
            return
        while True:
            try:
                state, status = observe(root, run_start)
                if status:
                    send_notification(registry, project, notification(state, status))
                    write_json(marker, {"run_start": run_start, "status": status,
                                        "notified_at": time.time()})
                    print(f"Notification acknowledged: {status}", flush=True)
                    return
            except Exception as exc:
                # Exception strings from HTTP clients can expose bot tokens.
                print(f"Watcher retry after {type(exc).__name__}", flush=True)
            time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(ARTIFACT_ROOT))
    parser.add_argument("--run-start", type=float, required=True)
    parser.add_argument("--registry", type=Path, default=Path(__file__).resolve().parents[3] / "arkhip/run/projects.yaml")
    parser.add_argument("--project", default="gigaevo-core")
    parser.add_argument("--interval", type=float, default=30)
    args = parser.parse_args()
    watch(args.root, args.run_start, args.registry, args.project, args.interval)


if __name__ == "__main__":
    main()
