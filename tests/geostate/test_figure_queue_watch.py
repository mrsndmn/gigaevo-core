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
