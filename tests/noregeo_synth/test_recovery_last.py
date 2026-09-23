import copy

import pytest

from experiments.noregeo_synth import recovery_eval as base
from experiments.noregeo_synth import recovery_last as ev
from experiments.noregeo_synth import run_recovery_eval as launch
from experiments.noregeo_synth.dataset import Task


def image(url):
    return {"type": "image_url", "image_url": {"url": url}}


def count_images(messages):
    return sum(
        x.get("type") == "image_url"
        for m in messages
        if isinstance(m["content"], list)
        for x in m["content"]
    )


def test_last_image_only_without_losing_text_or_mutating_trace():
    messages = [
        {"role": "system", "content": "rules"},
        {
            "role": "user",
            "content": [{"type": "text", "text": "step1 state"}, image("old")],
        },
        {"role": "assistant", "content": "old code"},
        {
            "role": "user",
            "content": [{"type": "text", "text": "step2 state"}, image("latest")],
        },
        {"role": "assistant", "content": "broken code"},
        {"role": "user", "content": [{"type": "text", "text": "failed; rolled back"}]},
    ]
    original = copy.deepcopy(messages)
    filtered = base.context_for(messages, "canvas_last")
    assert messages == original
    assert count_images(filtered) == 1
    assert filtered[3]["content"][-1] == image("latest")
    assert filtered[1]["content"] == [{"type": "text", "text": "step1 state"}]
    assert filtered[2:3] == messages[2:3]
    assert filtered[4:] == messages[4:]
    assert base.context_for(messages, "canvas") == original
    assert base.context_for(messages, "history") == original


def test_zero_images_and_replacement_frame():
    messages = [{"role": "user", "content": "start"}]
    assert base.context_for(messages, "canvas_last") == messages
    messages.extend(
        [
            {"role": "user", "content": [image("old")]},
            {
                "role": "user",
                "content": [{"type": "text", "text": "replaced"}, image("new")],
            },
        ]
    )
    assert count_images(base.context_for(messages, "canvas_last")) == 1
    assert base.context_for(messages, "canvas_last")[-1]["content"][-1] == image("new")


def raw(code, action="append"):
    return {
        "error": None,
        "content": f"ACTION: {action}\nTITLE: step\nREASONING: reason\n```python\n{code}\n```",
    }


@pytest.mark.parametrize(
    "arm,expected", [("canvas_last", [1, 1, 1]), ("canvas", [1, 2, 2])]
)
def test_action_and_final_requests_apply_retention(
    tmp_path, monkeypatch, arm, expected
):
    task = Task(
        "center_coordinates",
        "unused",
        {
            "qid": 1,
            "question": "A=(0,0), B=(2,2). Midpoint?",
            "points": {"A": [0, 0], "B": [2, 2]},
            "schema_type": "point",
            "answer": {"point": [1, 1]},
        },
    )
    replies = iter(
        [
            raw('mark("M", [1,1])\nstate["answer"]=[1,1]'),
            raw("pass", "finish"),
            {"error": None, "content": '{"final_answer":{"x":1,"y":1}}'},
        ]
    )
    captured = []

    def request(client, messages, *args, **kwargs):
        captured.append(copy.deepcopy(messages))
        return next(replies)

    monkeypatch.setattr(base, "_client", lambda cfg: (None, None))
    monkeypatch.setattr(base, "request", request)
    result = base.rollout(task, "block", arm, tmp_path, raw('mark("A",P["A"])'))
    assert result["correct"]
    assert [count_images(m) for m in captured] == expected
    assert all('mark("A"' in str(m) for m in captured)


def test_launcher_last_uses_correct_module_and_tls():
    payload = launch.payload(
        "commit",
        "/tmp/out",
        "/tmp/env",
        ev.SPEC,
        "experiments.noregeo_synth.recovery_last",
    )["job"]
    assert "recovery_last" in payload["script"]
    assert payload["description"].endswith(ev.SPEC.name)
    assert payload["environment"]["variables"]["SSL_CERT_FILE"].endswith(
        "tls-ca-bundle.pem"
    )


def test_report_counts_shared_first_but_physical_usage_only_new_calls(tmp_path):
    first = {
        "content": "cached",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    continuation = {
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
    }

    def record(arm):
        return {
            "arm": arm,
            "task_id": "a:1",
            "question": "test",
            "gold": 1,
            "first": first,
            "status": "completed",
            "correct": True,
            "events": [{"raw": first}],
            "final_raw": continuation,
        }

    base.write(
        tmp_path / "inputs.json",
        {
            "tasks": [{}],
            "baselines": [record(a) for a in base.ARMS],
            "first": {"a:1": first},
        },
    )
    base.write(tmp_path / "tasks/a_1/canvas_last/record.json", record("canvas_last"))
    base.write(tmp_path / "tasks/a_1/canvas_last/answer/result.json", continuation)
    result = ev.report(tmp_path)
    assert result["additional_tokens"] == 30
    assert result["arms"]["canvas_last"]["total_tokens_including_shared_first"] == 45
    assert (tmp_path / "analysis.html").exists()


def test_report_rejects_multiple_images(tmp_path):
    base.write(tmp_path / "inputs.json", {"tasks": [{}], "baselines": [], "first": {}})
    base.write(
        tmp_path / "tasks/a_1/canvas_last/call_02/request.json",
        {"messages": [{"role": "user", "content": [image("one"), image("two")]}]},
    )
    with pytest.raises(ValueError, match="multiple images"):
        ev.report(tmp_path)
