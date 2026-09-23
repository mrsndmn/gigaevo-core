import pytest

from experiments.noregeo_synth import recovery_eval as ev
from experiments.noregeo_synth.dataset import Task


def task():
    return Task(
        "center_coordinates",
        "unused",
        {
            "qid": 1,
            "question": "Points A=(0, 0), B=(2, 2). Find the midpoint M of segment AB.",
            "answer": {"point": [1, 1]},
            "schema_type": "point",
            "points": {"A": [0, 0], "B": [2, 2]},
        },
    )


def test_block_parser_preserves_quotes_and_rejects_truncation():
    text = 'ACTION: append\nTITLE: Test\nREASONING: Compute\n```python\nx = P["A"]\n```'
    assert ev.parse_step(text, "block", True)["code"] == 'x = P["A"]'
    with pytest.raises(SyntaxError):
        ev.parse_step(text.replace('P["A"]', "P["), "block", True)
    with pytest.raises(ValueError):
        ev.parse_step(text + "\nextra", "block", True)


def test_transaction_rolls_back_partial_mutation(tmp_path):
    initial = {"title": "initial", "code": 'state["x"] = 1'}
    broken = {"title": "broken", "code": 'state["x"] = 999\nraise ValueError("fail")'}
    failed = ev.execute(task(), [initial, broken], tmp_path / "failed")
    assert not failed["ok"]
    repaired = {"title": "repair", "code": 'state["answer"] = state["x"]\nprint("ok")'}
    success = ev.execute(task(), [initial, repaired], tmp_path / "repaired")
    assert success["ok"], success
    assert success["state"]["answer"] == 1
    assert success["stdout"].strip() == "ok"


def test_observation_differs_only_by_image(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"png")
    result = {"ok": True, "state": {"x": 1}, "image": str(image), "gold": "secret"}
    h = ev.feedback(result, "history", "next")
    c = ev.feedback(result, "canvas", "next")
    assert h["content"] == c["content"][:1]
    assert "secret" not in str(c)
    assert len(c["content"]) == 2


def test_gate_does_not_use_answer_correctness():
    records = [
        {
            "status": "completed",
            "events": [],
            "correct": False,
            "construction_ready": True,
        }
        for _ in range(30)
    ]
    assert ev.pilot_gate(records)["pass"]
    records[0]["events"] = [{"execution": {"stage": "format"}}]
    assert not ev.pilot_gate(records)["pass"]


def test_saved_request_does_not_call_endpoint(tmp_path):
    ev.write(tmp_path / "result.json", {"content": "cached"})
    assert ev.request(None, [], {}, "block", tmp_path, "salt")["content"] == "cached"


def test_incomplete_request_refuses_reroll(tmp_path):
    ev.write(tmp_path / "raw_00.json", {})
    with pytest.raises(RuntimeError, match="incomplete request"):
        ev.request(None, [], {}, "block", tmp_path, "salt")


def raw(code, action="append"):
    return {
        "error": None,
        "content": "ACTION: "
        + action
        + "\nTITLE: test\nREASONING: test\n```python\n"
        + code
        + "\n```",
    }


def test_rollout_repairs_and_finishes_without_fixed_four_steps(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "_client", lambda cfg: (None, None))
    calls = iter(
        [
            raw('state["answer"] = [1, 1]\nmark("M", [1, 1])'),
            raw("pass", "finish"),
            {"error": None, "content": '{"final_answer":{"x":1,"y":1}}'},
        ]
    )
    monkeypatch.setattr(ev, "request", lambda *a, **kw: next(calls))
    result = ev.rollout(
        task(),
        "block",
        "history",
        tmp_path,
        raw('state["answer"] = 99\nraise ValueError("bad")'),
    )
    assert result["status"] == "completed"
    assert result["correct"]
    assert result["committed"] == 1
    assert result["failures"] == 1
    assert len(result["steps"]) == 1


def test_bounded_repairs(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "_client", lambda cfg: (None, None))
    bad = raw("x = P[")
    monkeypatch.setattr(ev, "request", lambda *a, **kw: bad)
    result = ev.rollout(task(), "block", "history", tmp_path, bad)
    assert result["status"] == "execution_failed"
    assert len(result["events"]) == ev.SPEC.repair_budget + 1


def test_report_without_inference(tmp_path):
    ev.write(tmp_path / "inputs.json", {"cases": [], "tasks": [], "injection": []})
    ev.report(tmp_path)
    assert "NoReGeo" in (tmp_path / "analysis.html").read_text()


def test_gold_never_affects_policy_environment():
    from dataclasses import replace

    original = task()
    changed = replace(
        original,
        row={
            **original.row,
            "answer": {"point": [99999, -99999]},
            "image_answer": {"point": [-1234, 4567]},
        },
    )
    for fmt in ev.FORMATS:
        assert ev.messages_for(original, fmt) == ev.messages_for(changed, fmt)
    assert ev.preamble(original) == ev.preamble(changed)


def test_interrupted_inflight_request_cannot_reroll(tmp_path):
    from types import SimpleNamespace

    count = []

    def interrupted(**kwargs):
        count.append(1)
        raise KeyboardInterrupt()

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=interrupted))
    )
    with pytest.raises(KeyboardInterrupt):
        ev.request(client, [], {}, "block", tmp_path, "same")
    with pytest.raises(RuntimeError, match="incomplete request"):
        ev.request(client, [], {}, "block", tmp_path, "same")
    assert len(count) == 1


def test_sdk_retries_disabled(monkeypatch):
    from types import SimpleNamespace

    options = []
    client = SimpleNamespace(
        with_options=lambda **kwargs: options.append(kwargs) or "client"
    )
    monkeypatch.setattr(ev, "_base_client", lambda cfg: (client, "credentials"))
    assert ev._client(ev.CFG) == ("client", "credentials")
    assert options == [{"max_retries": 0}]


def test_empty_construction_cannot_finish_or_pass_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "_client", lambda cfg: (None, None))
    monkeypatch.setattr(ev, "request", lambda *a, **kw: raw("pass", "finish"))
    result = ev.rollout(task(), "block", "history", tmp_path, raw("pass"))
    assert result["status"] == "execution_failed"
    assert not ev.pilot_gate([result] * 30)["pass"]
    assert not ev.pilot_gate(
        [{"status": "completed", "events": [], "correct": True}] * 30
    )["pass"]


def test_incorrect_but_visible_construction_is_technically_ready(tmp_path):
    result = ev.execute(
        task(),
        [{"title": "wrong", "code": 'state["answer"] = [2, 2]\nmark("M", [2, 2])'}],
        tmp_path,
    )
    assert ev.construction_ready(result)
