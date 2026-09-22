"""Feedback isolation, deterministic replay, and end-to-end local execution."""

from dataclasses import replace
import json

import pytest

from experiments.noregeo_synth import feedback as fb
from experiments.noregeo_synth.dataset import Task


@pytest.fixture
def task():
    return Task(
        "midpoint",
        "unused",
        {
            "qid": 1,
            "question": "Find the midpoint of A=(0,0) and B=(2,2).",
            "schema_type": "point",
            "points": {"A": [0, 0], "B": [2, 2]},
            "answer": [1, 1],
        },
    )


STEPS = [
    {
        "title": "Given segment",
        "reasoning": "Connect the endpoints.",
        "code": 'ax.plot([0,2], [0,2]); mark("A", P["A"]); mark("B", P["B"])',
    },
    {
        "title": "Midpoint",
        "reasoning": "Average the endpoints.",
        "code": 'state["answer"] = (P["A"] + P["B"]) / 2; mark("M", state["answer"])',
    },
]


def config(mode):
    return fb.FeedbackExperiment(
        mode, mode, replace(fb._BASE, min_steps=2, max_steps=2)
    )


def test_initial_prompt_does_not_depend_on_gold(task):
    changed = replace(task, row={**task.row, "answer": [9876, 5432]})
    assert fb.initial_messages(task, fb._BASE, "PREAMBLE") == fb.initial_messages(
        changed, replace(fb._BASE, reveal_answer=True), "PREAMBLE"
    )


def test_only_image_differs_in_observation(tmp_path):
    image = tmp_path / "x.png"
    image.write_bytes(b"image")
    status = fb.observation(image, "status", "Next")
    canvas = fb.observation(image, "image", "Next")
    assert canvas["content"][:1] == status["content"]
    assert canvas["content"][1]["type"] == "image_url"
    assert "state" not in json.dumps(status)


@pytest.mark.parametrize("mode", ["status", "image"])
def test_local_rollout_and_final_observation(task, tmp_path, mode):
    requests = []

    def call(client, cfg, messages, schema, salt):
        requests.append(json.loads(json.dumps(messages)))
        n = len(requests)
        return (STEPS[n - 1] if n <= 2 else {"final_answer": {"x": 1, "y": 1}}, "", {})

    out = tmp_path / mode
    result = fb.rollout(task, config(mode), out, None, call=call)
    assert result["status"] == "completed"
    assert result["accepted"], result["checks"]
    assert len(requests) == 3
    image_counts = [json.dumps(r).count('"image_url":') for r in requests]
    assert image_counts == ([0, 1, 2] if mode == "image" else [0, 0, 0])
    assert all("verified correct answer" not in json.dumps(r) for r in requests)
    assert (out / "trace.md").exists()
    assert (out / "contact_sheet.png").exists()
    assert json.loads((out / "record.json").read_text())["accepted"]
    with pytest.raises(FileExistsError):
        fb.rollout(task, config(mode), out, None, call=call)


def test_bad_code_stops_without_gold_feedback(task, tmp_path):
    def call(*args, **kwargs):
        return {"title": "bad", "reasoning": "bad", "code": "open('x')"}, "", {}

    out = tmp_path / "bad"
    result = fb.rollout(task, config("status"), out, None, call=call)
    assert result["status"] == "execution_failed"
    assert not result["accepted"]
    assert len(result["calls"]) == 1
    assert (out / "record.json").exists()


def test_call_failure_is_saved(task, tmp_path):
    def call(*args, **kwargs):
        raise RuntimeError("offline endpoint")

    out = tmp_path / "error"
    with pytest.raises(RuntimeError, match="offline endpoint"):
        fb.rollout(task, config("status"), out, None, call=call)
    record = json.loads((out / "record.json").read_text())
    assert record["status"] == "error"
    assert not record["accepted"]


def test_shared_first_step_is_executed_without_resampling(task, tmp_path):
    calls = []

    def call(client, cfg, messages, schema, salt):
        calls.append(messages.copy())
        return (
            STEPS[1] if len(calls) == 1 else {"final_answer": {"x": 1, "y": 1}},
            "",
            {},
        )

    result = fb.rollout(
        task, config("image"), tmp_path / "shared", None, call=call, first_step=STEPS[0]
    )
    assert result["shared_first_step"]
    assert result["steps"][0] == STEPS[0]
    assert len(calls) == 2
    assert calls[0][-1]["content"][1]["type"] == "image_url"
    assert result["accepted"]


def test_worker_preserves_unrendered_state_and_frames(task, tmp_path):
    from experiments.noregeo_synth.feedback_execute import DrawingSession

    cfg = config("status").generation
    session = DrawingSession(fb.build_preamble(task.row, cfg), tmp_path, cfg, None)
    try:
        first = session.step(
            {
                "title": "one",
                "code": 'state["answer"] = float(np.random.random()); mark("A", P["A"])',
            }
        )
        value = first.state_answer
        original = (tmp_path / "step_01.png").read_bytes()
        second = session.step(
            {"title": "two", "code": 'state["answer"] += 1; mark("B", P["B"])'}
        )
        assert second.state_answer == value + 1
        assert len(second.steps) == 2
        assert (tmp_path / "step_01.png").read_bytes() == original
    finally:
        session.close()
    assert session.process.poll() is not None


def test_worker_timeout_kills_process(task, tmp_path):
    from experiments.noregeo_synth.feedback_execute import DrawingSession

    cfg = config("status").generation
    session = DrawingSession(fb.build_preamble(task.row, cfg), tmp_path, cfg, None)
    session.cfg = replace(cfg, exec_timeout=0.1)
    try:
        with pytest.raises(TimeoutError):
            session.step({"title": "loop", "code": "while True: pass"})
        assert session.process.poll() is not None
    finally:
        session.close()


def test_cli_continues_other_arms_and_tasks(task, tmp_path, monkeypatch):
    second = replace(task, row={**task.row, "qid": 2})
    monkeypatch.setattr(fb.ds, "load_tasks", lambda: [task, second])
    monkeypatch.setattr(fb, "_client", lambda cfg: (None, None))
    monkeypatch.setattr(fb, "check_model_is_served", lambda *args: None)
    seen = []

    def rollout(task, experiment, out, client, first_step):
        seen.append((task.task_id, experiment.name, first_step))
        out.mkdir(parents=True)
        record = {
            "task_id": task.task_id,
            "arm": experiment.name,
            "steps": [STEPS[0]],
            "error": None,
            "accepted": True,
            "status": "completed",
        }
        if len(seen) == 1:
            record.update(error="unusable completion", accepted=False, status="error")
        fb.write_json(out / "record.json", record)
        if len(seen) == 1:
            raise RuntimeError("unusable completion")
        return record

    monkeypatch.setattr(fb, "rollout", rollout)
    assert (
        fb.main(
            [
                "--task",
                task.task_id,
                "--task",
                second.task_id,
                "--out",
                str(tmp_path / "run"),
            ]
        )
        == 1
    )
    assert len(seen) == 4
    assert seen[1][2] == STEPS[0]
    assert seen[2][2] is None
