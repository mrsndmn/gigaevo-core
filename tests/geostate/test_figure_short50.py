"""The figure pilot must use precisely the saved text problems and copies."""

import json
from dataclasses import replace

import pytest

from experiments.geostate import run_figure_short50 as pilot
from experiments.geostate import consistency as c
from experiments.geostate.configs import config_named


def test_pilot_is_paired_idempotent_and_rejects_changed_source(tmp_path, monkeypatch):
    source = tmp_path / "source"
    c.prepare(config_named("e0-synthetic-text"), source)
    monkeypatch.setattr(pilot, "FIGURE_COHORTS", {
        name: replace(cohort, source=str(source)) for name, cohort in pilot.FIGURE_COHORTS.items()
    })
    out = tmp_path / "pilot"
    pilot.freeze(out)
    prepared = c.prepare(config_named("e0-synthetic-figure-short50"), out)
    generic = c.prepare(config_named("e0-synthetic-figure-short50"), tmp_path / "generic")
    assert [(p.id, len(cs)) for p, cs in generic] == [(p.id, len(cs)) for p, cs in prepared]
    assert len(prepared) == 50
    assert all(p.bucket == "1-3" and len(cs) == 8 for p, cs in prepared)
    next_cohort = c.prepare(config_named("e0-synthetic-figure-short-next50"), tmp_path / "next")
    next_ids = {p.id for p, _ in next_cohort}
    assert len(next_cohort) == 50
    assert all(p.bucket == "1-3" and len(cs) == 8 for p, cs in next_cohort)
    assert not next_ids.intersection(p.id for p, _ in prepared)
    source_copies = [json.loads(x) for x in (source / "copies.jsonl").read_text().splitlines()]
    selected = {p.id for p, _ in prepared}
    source_short_ids = [p.id for p, _ in c.prepare(config_named("e0-synthetic-text"), source) if p.bucket == "1-3"]
    assert selected.union(next_ids) == set(source_short_ids)
    for name, bucket in [("e0-synthetic-figure-medium", "4-9"), ("e0-synthetic-figure-long", "10+")]:
        rows = c.prepare(config_named(name), tmp_path / name)
        assert len(rows) == 100
        assert all(p.bucket == bucket for p, _ in rows)
        assert {p.id for p, _ in rows}.isdisjoint(selected | next_ids)
        assert config_named(name).workers == 8
    # The aux cohort selects every source row, without synthetic bucket filtering.
    aux = c.prepare(config_named("e0-aux-figure-paired"), tmp_path / "aux")
    assert {p.id for p, _ in aux} == {p.id for p, _ in c.prepare(config_named("e0-synthetic-text"), source)}
    assert config_named("e0-aux-figure-paired").workers == 8
    assert (out / "copies.jsonl").read_text().splitlines() == [
        json.dumps(x) for x in source_copies if x["problem_id"] in selected
    ]
    pilot.freeze(out)
    rows = (source / "copies.jsonl").read_text().splitlines()
    first = json.loads(rows[0])
    first["copy"]["kind"] = "changed"
    rows[0] = json.dumps(first)
    (source / "copies.jsonl").write_text("\n".join(rows) + "\n")
    # Read-only summaries must not depend on the current source or protocol.
    assert len(c.prepare(config_named("e0-synthetic-figure-short50"), out, strict=False)) == 50
    with pytest.raises(ValueError, match="differs"):
        pilot.freeze(out)


@pytest.mark.parametrize("name", list(pilot.FIGURE_COHORTS))
def test_pilot_rejects_mls_submission(name):
    from experiments.geostate.run_e0_jobs import main

    with pytest.raises(SystemExit) as error:
        main(["--config", name, "--dry"])
    assert error.value.code == 2


def test_queue_serializes_cohorts_and_continues_after_failed_requests(tmp_path, monkeypatch):
    from experiments.geostate import run_figure_queue as queue

    names = ("e0-synthetic-figure-short50", "e0-synthetic-figure-medium")
    monkeypatch.setattr(queue, "FIGURE_QUEUE", names)
    events = []
    monkeypatch.setattr(queue, "freeze", lambda out, name: events.append(("freeze", name)))
    monkeypatch.setattr(queue, "recover_errors", lambda out: [])

    def execute(name, out):
        events.append(("execute", name))
        assert config_named(name).workers == 8
        return 3 if name == names[0] else 0

    monkeypatch.setattr(queue, "execute", execute)
    assert queue.run_queue(tmp_path) == 1
    assert events == [(action, name) for name in names for action in ("freeze", "execute")]
    state = json.loads((tmp_path / "figure-queue-state.json").read_text())
    assert state["status"] == "finished_with_errors"
    assert state["cohorts"][names[0]]["status"] == "needs_retry"
    assert state["cohorts"][names[1]]["status"] == "complete"


def test_queue_refuses_a_second_controller(tmp_path):
    import fcntl
    from experiments.geostate.run_figure_queue import run_queue

    with (tmp_path / ".figure-queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            run_queue(tmp_path)
