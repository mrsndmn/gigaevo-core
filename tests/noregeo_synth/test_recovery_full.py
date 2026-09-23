from dataclasses import replace

import pytest

from experiments.noregeo_synth import recovery_full as ev
from experiments.noregeo_synth import run_recovery_eval as launch
from experiments.noregeo_synth.feedback_eval import read, write


def test_recover_received_raw_without_generation(tmp_path):
    write(
        tmp_path / "raw_00.json",
        {
            "choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 42},
        },
    )
    ev.recover_raw(tmp_path)
    assert read(tmp_path / "result.json")["usage"]["total_tokens"] == 42
    assert read(tmp_path / "result.json")["content"] == "answer"


def test_unknown_inflight_is_not_rerolled(tmp_path):
    write(tmp_path / "record.json", {"status": "running"})
    write(tmp_path / "call_2/request.json", {"messages": []})
    with pytest.raises(RuntimeError, match="unresolved"):
        ev.resume_arm(tmp_path)
    assert (tmp_path / "record.json").exists()


def test_terminal_failure_is_preserved(tmp_path):
    write(tmp_path / "record.json", {"status": "execution_failed"})
    ev.resume_arm(tmp_path)
    assert read(tmp_path / "record.json")["status"] == "execution_failed"


def test_cached_request_checks_context(tmp_path):
    write(tmp_path / "request.json", {"messages": ["a"]})
    write(tmp_path / "result.json", {"content": "ok", "error": None})
    assert (
        ev.full_request(None, ["a"], {}, "block", tmp_path, "salt")["content"] == "ok"
    )
    with pytest.raises(RuntimeError, match="context"):
        ev.full_request(None, ["b"], {}, "block", tmp_path, "salt")


def test_transport_errors_archived_success_not_rerolled(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "SPEC", replace(ev.SPEC, transport_backoff_seconds=0))

    class Client:
        def close(self):
            pass

    monkeypatch.setattr(ev.base, "_client", lambda cfg: (Client(), None))
    calls = []

    def request(client, messages, schema, fmt, directory, salt):
        calls.append(1)
        result = (
            {"error": "APIConnectionError: network"}
            if len(calls) == 1
            else {"error": None, "content": "ok"}
        )
        write(directory / "request.json", {"messages": messages})
        write(directory / "result.json", result)
        return result

    monkeypatch.setattr(ev.base, "request", request)
    directory = tmp_path / "call"
    assert ev.full_request(None, [], {}, "block", directory, "s")["content"] == "ok"
    assert ev.full_request(None, [], {}, "block", directory, "s")["content"] == "ok"
    assert len(calls) == 2
    assert len(list((tmp_path / "transport_attempts").glob("*/*/result.json"))) == 1


def test_all_arm_orders_are_permutations():
    orders = {ev.arm_order(str(i)) for i in range(100)}
    assert len(orders) == 6
    assert all(set(order) == set(ev.ARMS) for order in orders)


def test_full_launcher_tls():
    job = launch.payload(
        "commit",
        "/tmp/out",
        "/tmp/env",
        ev.SPEC,
        "experiments.noregeo_synth.recovery_full",
    )["job"]
    assert "recovery_full" in job["script"]
    assert job["environment"]["variables"]["SSL_CERT_FILE"].endswith(
        "tls-ca-bundle.pem"
    )


def test_report_primary_all_shared_tokens_and_mismatched_first(tmp_path):
    tasks = [
        {"stem": "valid", "row": {"qid": 1, "question": "test"}},
        {"stem": "broken", "row": {"qid": 2, "question": "known issue"}},
    ]
    write(tmp_path / "inputs.json", {"tasks": tasks, "issues": {"broken": "defect"}})
    first = {
        "content": "first",
        "usage": {"total_tokens": 15, "prompt_tokens": 10, "completion_tokens": 5},
    }
    for task in tasks:
        taskid = ev.extra.task_id(task)
        root = tmp_path / "tasks" / taskid.replace(":", "_")
        write(root / "first/result.json", first)
        for arm in ev.ARMS:
            write(
                root / arm / "record.json",
                {
                    "task_id": taskid,
                    "arm": arm,
                    "first": first,
                    "question": "test",
                    "gold": 1,
                    "status": "completed",
                    "correct": arm != "history",
                    "events": [{"call": 1, "raw": first}],
                    "final_answer": 1,
                },
            )
    result = ev.report(tmp_path)
    assert result["primary"]["arms"]["history"]["wrong_answer"] == 1
    assert result["all"]["arms"]["history"]["wrong_answer"] == 2
    assert result["accounted_tokens"] == 30
    assert result["primary"]["paired"]["history vs canvas"]["right_only"] == 1
    assert (tmp_path / "tasks/valid_1/analysis.html").exists()
    path = tmp_path / "tasks/valid_1/history/record.json"
    record = read(path)
    record["first"] = {}
    write(path, record)
    with pytest.raises(ValueError, match="canonical"):
        ev.report(tmp_path)


@pytest.mark.parametrize(
    "error", ["AuthenticationError: expired", "APITimeoutError: timeout"]
)
def test_resume_retries_saved_infrastructure_rejections(tmp_path, monkeypatch, error):
    monkeypatch.setattr(ev, "SPEC", replace(ev.SPEC, transport_backoff_seconds=0))
    directory = tmp_path / "call"
    write(directory / "request.json", {"messages": []})
    write(directory / "result.json", {"error": error})

    class Client:
        def close(self):
            pass

    monkeypatch.setattr(ev.base, "_client", lambda cfg: (Client(), None))
    calls = []

    def request(*args):
        calls.append(1)
        return {"error": None, "content": "renewed"}

    monkeypatch.setattr(ev.base, "request", request)
    assert (
        ev.full_request(None, [], {}, "block", directory, "s")["content"] == "renewed"
    )
    assert len(calls) == 1


def test_report_tolerates_archived_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "record.json"
    write(path, {"status": "running"})
    original = ev.read

    def racing_read(p):
        p.unlink()
        return original(p)

    monkeypatch.setattr(ev, "read", racing_read)
    assert ev.read_available([path]) == []


def test_format_failure_trace_contains_raw_final_and_parser_error(tmp_path):
    write(
        tmp_path / "history/record.json",
        {
            "task_id": "a:1",
            "question": "question",
            "gold": 1,
            "arm": "history",
            "status": "format_error",
            "correct": False,
            "events": [],
            "final_raw": {
                "content": "malformed final <evidence>",
                "finish_reason": "length",
            },
            "error": "missing final_answer",
        },
    )
    ev.trace(tmp_path)
    html = (tmp_path / "analysis.html").read_text()
    assert "malformed final &lt;evidence&gt;" in html
    assert "missing final_answer" in html
    assert "length" in html
