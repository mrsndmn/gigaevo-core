"""Frozen cohorts, fixed denominators and resumable pair identity."""

from dataclasses import asdict, replace
import threading

import pytest

from experiments.noregeo_synth import feedback_eval as ev
from experiments.noregeo_synth import run_feedback_eval as launch
from experiments.noregeo_synth.dataset import Task


def task(qid=1):
    return Task(
        "midpoint",
        "unused",
        {"qid": qid, "question": "Midpoint?", "answer": [1, 1], "schema_type": "point"},
    )


def historical(t, correct=False):
    return {
        "task_id": t.task_id,
        "arm": "text",
        "question": t.row["question"],
        "golden_answer": t.row["answer"],
        "ok": correct,
    }


def test_cohort_is_not_filtered_by_outcome():
    tasks = [task(1), task(2)]
    rows = [historical(tasks[0], False), historical(tasks[1], True)]
    spec = replace(ev.SPEC, task_count=2)
    assert ev.cohort(rows, tasks, spec) == tasks
    with pytest.raises(ValueError, match="size/uniqueness"):
        ev.cohort(rows + rows[:1], tasks, spec)
    rows[0]["question"] = "changed"
    with pytest.raises(ValueError, match="benchmark changed"):
        ev.cohort(rows, tasks, spec)


def test_metrics_preserve_missing_and_failed_cases(tmp_path):
    tasks = [task(1), task(2)]
    ev.write(
        tmp_path / "inputs.json",
        {
            "tasks": [asdict(t) for t in tasks],
            "historical": [historical(t, True) for t in tasks],
        },
    )
    h = {"status": "execution_failed", "accepted": False, "steps": [{"code": "same"}]}
    c = {
        "status": "completed",
        "checks": {"answer_ok": True},
        "accepted": True,
        "steps": [{"code": "same"}],
    }
    ev.write(ev.arm_directory(tmp_path, tasks[0].task_id, "history") / "record.json", h)
    ev.write(ev.arm_directory(tmp_path, tasks[0].task_id, "canvas") / "record.json", c)
    ev.write(
        ev.task_directory(tmp_path, tasks[0].task_id) / "text.json",
        {"error": "timeout", "ok": False},
    )
    data = ev.comparison(tmp_path)
    assert data["rows"][0] == {
        "arm": "text",
        "planned": 2,
        "finished": 0,
        "correct": 0,
        "errors": 1,
        "accepted": None,
    }
    assert data["rows"][1]["finished"] == 1
    assert data["paired"]["canvas_only"] == 1
    assert data["paired"]["n"] == 1
    assert all(r["n"] <= 1 for r in data["matched"])


def test_exact_paired_test():
    h = {"checks": {"answer_ok": False}}
    c = {"checks": {"answer_ok": True}}
    assert ev.paired_counts([(h, c)] * 6)["mcnemar_exact_p"] == 0.03125
    assert ev.paired_counts([])["mcnemar_exact_p"] == 1


@pytest.mark.parametrize("has_history_first", [True, False])
def test_resume_reuses_first_action_and_skips_completed(
    tmp_path, monkeypatch, has_history_first
):
    t = task()
    directory = ev.task_directory(tmp_path, t.task_id)
    ev.write(directory / "text.json", {"ok": True, "error": None})
    first = {"title": "first", "reasoning": "same", "code": "pass"}
    ev.write(
        ev.arm_directory(tmp_path, t.task_id, "history") / "record.json",
        {
            "status": "error",
            "steps": [first] if has_history_first else [],
            "error": "completion",
        },
    )
    ev.write(
        ev.arm_directory(tmp_path, t.task_id, "canvas") / "record.json",
        {"status": "completed", "steps": [first]},
    )
    monkeypatch.setattr(ev, "_client", lambda cfg: (None, None))
    monkeypatch.setattr(ev, "build_report", lambda *args: None)
    seen = []

    def rollout(task, arm, destination, client, first_step):
        seen.append((arm.name, first_step))
        r = {"status": "completed", "steps": [first]}
        ev.write(destination / "record.json", r)
        return r

    monkeypatch.setattr(ev, "rollout", rollout)
    ev.evaluate_task(t, tmp_path, threading.Event())
    assert seen == [("history", first)]
    assert ev.read(directory / "first_step.json") == first
    assert list((directory / "attempts").glob("history-*"))


def test_job_manifest_does_not_contain_token_and_maps_paths():
    p = launch.payload(
        "abc",
        "/mnt/virtual_ai0001053-00054_SR004-nfs2/d.tarasov/output",
        "/mnt/virtual_ai0001053-00054_SR004-nfs2/d.tarasov/graphify/.env",
    )
    job = p["job"]
    assert job["resource"]["processes"] == 1
    assert "AIRI_API_KEY" not in job["environment"]["variables"]
    assert job["environment"]["variables"]["AIRI_ENV_FILE"].startswith(
        "/workspace-SR004.nfs2/"
    )
    assert "--run --out /workspace-SR004.nfs2/" in job["script"]
    assert launch.JOB_ID.fullmatch("lm-mpi-job-48fcb78d-e22e-4076-a28d-f92f4e1d6b0c")
    assert not launch.JOB_ID.fullmatch("lm-mpi-job-48fcb78d")


def test_concurrent_writers_use_distinct_temporaries(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    barrier = threading.Barrier(2)
    replace_file = Path.replace
    target = tmp_path / "comparison.json"

    def synchronized(self, destination):
        if destination == target:
            barrier.wait(timeout=5)
        return replace_file(self, destination)

    monkeypatch.setattr(Path, "replace", synchronized)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda i: ev.write(target, {"writer": i}), [1, 2]))
    assert ev.read(target)["writer"] in (1, 2)
    assert list(tmp_path.iterdir()) == [target]


def test_conflicting_first_actions_are_rejected(tmp_path):
    for arm in ("history", "canvas"):
        ev.write(
            ev.arm_directory(tmp_path, "midpoint:1", arm) / "record.json",
            {"status": "completed", "steps": [{"code": arm}]},
        )
    with pytest.raises(ValueError, match="conflicting first actions"):
        ev.recover_first_step(tmp_path, "midpoint:1")


def test_real_drawing_timeouts_count_as_terminal_wrong_pair(tmp_path, monkeypatch):
    from experiments.noregeo_synth import feedback as fb

    original_session = fb.DrawingSession

    def short_step_timeout(*args, **kwargs):
        session = original_session(*args, **kwargs)
        session.cfg = replace(session.cfg, exec_timeout=0.1)
        return session

    monkeypatch.setattr(fb, "DrawingSession", short_step_timeout)

    def call(*args, **kwargs):
        return (
            {"title": "loop", "reasoning": "loop", "code": "while True: pass"},
            "",
            {},
        )

    t = task()
    ev.write(
        tmp_path / "inputs.json", {"tasks": [asdict(t)], "historical": [historical(t)]}
    )
    for arm in fb.experiment_configs:
        record = fb.rollout(
            t, arm, ev.arm_directory(tmp_path, t.task_id, arm.name), None, call=call
        )
        assert record["status"] == "execution_failed"
        assert ev.finished(record, arm.name)
    metrics = ev.comparison(tmp_path)
    assert metrics["paired"]["n"] == 1
    assert metrics["paired"]["neither"] == 1
