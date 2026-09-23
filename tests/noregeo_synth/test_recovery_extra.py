from experiments.noregeo_synth import recovery_extra as ev
from experiments.noregeo_synth import run_recovery_eval as launcher


def task(stem, qid):
    return {"stem": stem, "row": {"qid": qid}}


def test_selection_excludes_old_and_prioritizes_new_stems():
    tasks = [task(s, i) for s in ["a", "b", "c"] for i in [1, 2, 3]]
    old = [task("a", 1)]
    chosen = ev.select_tasks(tasks, old, 3)
    assert chosen == [task("b", 1), task("c", 1), task("a", 2)]
    assert not ({ev.task_id(t) for t in old} & {ev.task_id(t) for t in chosen})


def record(arm):
    first = {"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
    last = {"usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}}
    return {
        "task_id": "a:1",
        "arm": arm,
        "status": "completed",
        "correct": True,
        "events": [{"raw": first}],
        "first": first,
        "final_raw": last,
    }


def test_token_dedup_including_partial_pairs():
    single = ev.group_metrics([record("history")], 1)
    paired = ev.group_metrics([record("history"), record("canvas")], 1)
    assert single["actual_tokens"] == 45
    assert paired["actual_tokens"] == 75
    assert paired["arms"]["canvas"]["prompt_tokens"] == 30
    assert paired["paired"]["both"] == 1


def test_launcher_payload_selects_correct_module_and_description():
    old = launcher.payload("commit", "/tmp/out", "/tmp/env")
    new = launcher.payload(
        "commit",
        "/tmp/out",
        "/tmp/env",
        ev.SPEC,
        "experiments.noregeo_synth.recovery_extra",
    )
    assert "recovery_extra" not in old["job"]["script"]
    assert "recovery_extra" in new["job"]["script"]
    assert new["job"]["description"].endswith(ev.SPEC.name)
    assert old["job"]["description"] != new["job"]["description"]


def test_shared_first_counted_once_before_event_checkpoint():
    h = record("history")
    c = record("canvas")
    for r in [h, c]:
        r["status"] = "running"
        r["correct"] = False
        r.pop("final_raw")
    c["events"] = []
    assert ev.group_metrics([h, c], 1)["actual_tokens"] == 15
    h["events"] = []
    assert ev.group_metrics([h, c], 1)["actual_tokens"] == 15
    assert ev.group_metrics([h], 1)["actual_tokens"] == 15


def test_connection_recovery_preserves_successful_calls(tmp_path, monkeypatch):
    from experiments.noregeo_synth.feedback_eval import read, write

    monkeypatch.setattr(launcher, "mls", lambda *a: {"status": "completed"})
    monkeypatch.setattr(ev, "report", lambda out: None)
    write(
        tmp_path / "submission.json",
        {"job_name": "lm-mpi-job-00000000-0000-0000-0000-000000000000"},
    )
    write(tmp_path / "complete.json", {"status": "attempted"})
    base = tmp_path / "tasks" / "a_1"
    good = base / "history" / "record.json"
    write(good, {"status": "completed"})
    bad = base / "canvas" / "record.json"
    write(bad, {"status": "request_error"})
    cached = base / "canvas" / "call_02" / "result.json"
    write(cached, {"error": None, "content": "successful response"})
    failed = base / "canvas" / "call_03"
    write(failed / "result.json", {"error": "APIConnectionError: Connection error."})
    before = good.read_bytes(), cached.read_bytes()
    archive = ev.recover_connections(tmp_path)
    assert not failed.exists() and not bad.exists()
    assert not (tmp_path / "complete.json").exists()
    assert (good.read_bytes(), cached.read_bytes()) == before
    assert (
        read(archive / "tasks" / "a_1" / "canvas" / "record.json")["status"]
        == "request_error"
    )


def test_connection_recovery_refuses_live_job(tmp_path, monkeypatch):
    import pytest

    from experiments.noregeo_synth.feedback_eval import write

    monkeypatch.setattr(launcher, "mls", lambda *a: {"status": "running"})
    write(tmp_path / "submission.json", {"job_name": "job"})
    with pytest.raises(RuntimeError, match="active"):
        ev.recover_connections(tmp_path)


def test_tls_bundle_keeps_verification(tmp_path):
    import ssl

    ev.write_tls_bundle(tmp_path)
    context = ssl.create_default_context(cafile=str(tmp_path / "tls-ca-bundle.pem"))
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname
    payload = launcher.payload(
        "commit",
        tmp_path,
        "/tmp/env",
        ev.SPEC,
        "experiments.noregeo_synth.recovery_extra",
    )
    assert payload["job"]["environment"]["variables"]["SSL_CERT_FILE"].endswith(
        "tls-ca-bundle.pem"
    )
