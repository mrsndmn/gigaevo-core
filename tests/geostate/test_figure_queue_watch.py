import json

import pytest

from experiments.geostate import watch_figure_queue as watcher


@pytest.mark.parametrize("state,locked,expected", [
    ({"started": 1, "status": "running"}, True, None),
    ({"started": 1, "status": "running"}, False, "interrupted"),
    ({"started": 1, "status": "complete"}, True, "complete"),
    ({"started": 1, "status": "finished_with_errors"}, False, "finished_with_errors"),
    ({"started": 2, "status": "complete"}, False, "superseded"),
])
def test_terminal_status(state, locked, expected):
    assert watcher.terminal_status(state, 1, locked) == expected


def test_watch_retries_and_deduplicates(tmp_path, monkeypatch):
    (tmp_path / "figure-queue-state.json").write_text(json.dumps({
        "started": 1, "status": "complete", "cohorts": {"short": {"status": "complete", "evaluated": 400}}
    }))
    sent = []
    def send(*args):
        sent.append(args[-1])
        if len(sent) == 1:
            raise RuntimeError("temporary delivery failure")
    monkeypatch.setattr(watcher, "send_notification", send)
    monkeypatch.setattr(watcher.time, "sleep", lambda _: None)
    watcher.watch(tmp_path, 1, tmp_path / "registry", "project")
    watcher.watch(tmp_path, 1, tmp_path / "registry", "project")
    assert len(sent) == 2
    assert "400" in sent[-1]
    assert json.loads((tmp_path / "figure-queue-watch-1.000000.json").read_text())["status"] == "complete"


def test_lock_detection(tmp_path):
    import fcntl
    assert not watcher.queue_locked(tmp_path)
    with (tmp_path / ".figure-queue.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert watcher.queue_locked(tmp_path)


def test_transport_failure_does_not_log_token(tmp_path, monkeypatch, caplog):
    import agenttg.api
    import requests
    registry = tmp_path / "projects.yaml"
    registry.write_text('projects:\n  project:\n    telegram_chat_id: "123"\n    telegram_thread_id: 42\n')
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "FAKE_SECRET_TOKEN")
    def fail(*args, **kwargs):
        raise requests.ConnectionError('/botFAKE_SECRET_TOKEN/sendMessage')
    monkeypatch.setattr(agenttg.api, "_request_with_retry", fail)
    previous = agenttg.api.logger.disabled
    with pytest.raises(RuntimeError, match="acknowledge"):
        watcher.send_notification(registry, "project", "hello")
    assert "FAKE_SECRET_TOKEN" not in caplog.text
    assert agenttg.api.logger.disabled == previous


@pytest.mark.parametrize("status,start,expected", [
    ("complete", 1, "complete"),
    ("finished_with_errors", 1, "finished_with_errors"),
    ("running", 2, "superseded"),
])
def test_observe_reads_state_after_lock_acquisition(tmp_path, monkeypatch, status, start, expected):
    state_path = tmp_path / "figure-queue-state.json"
    state_path.write_text(json.dumps({"started": 1, "status": "running"}))
    flock = watcher.fcntl.flock
    def finish_at_lock(handle, operation):
        flock(handle, operation)
        state_path.write_text(json.dumps({"started": start, "status": status}))
    monkeypatch.setattr(watcher.fcntl, "flock", finish_at_lock)
    _, actual = watcher.observe(tmp_path, 1)
    assert actual == expected
