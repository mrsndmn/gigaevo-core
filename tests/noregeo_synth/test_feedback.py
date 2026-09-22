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


def test_replay_mismatch_aborts(task, tmp_path, monkeypatch):
    real_run = fb.run_steps

    def changed_run(preamble, steps, out, cfg, target):
        result = real_run(preamble, steps, out, cfg, target)
        if len(steps) == 2:
            from pathlib import Path

            Path(result.steps[0].image).write_bytes(b"changed")
        return result

    monkeypatch.setattr(fb, "run_steps", changed_run)
    count = 0

    def call(*args, **kwargs):
        nonlocal count
        result = STEPS[count]
        count += 1
        return result, "", {}

    with pytest.raises(RuntimeError, match="replay changed"):
        fb.rollout(task, config("status"), tmp_path / "mismatch", None, call=call)
    assert count == 2


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
