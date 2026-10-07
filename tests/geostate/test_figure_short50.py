"""The figure pilot must use precisely the saved text problems and copies."""

import json

import pytest

from experiments.geostate import run_figure_short50 as pilot
from experiments.geostate import consistency as c
from experiments.geostate.configs import config_named


def test_pilot_is_paired_idempotent_and_rejects_changed_source(tmp_path, monkeypatch):
    source = tmp_path / "source"
    c.prepare(config_named("e0-synthetic-text"), source)
    monkeypatch.setattr(pilot, "FIGURE_SHORT50_SOURCE", str(source))
    out = tmp_path / "pilot"
    pilot.freeze(out)
    prepared = c.prepare(config_named("e0-synthetic-figure-short50"), out)
    generic = c.prepare(config_named("e0-synthetic-figure-short50"), tmp_path / "generic")
    assert [(p.id, len(cs)) for p, cs in generic] == [(p.id, len(cs)) for p, cs in prepared]
    assert len(prepared) == 50
    assert all(p.bucket == "1-3" and len(cs) == 8 for p, cs in prepared)
    source_copies = [json.loads(x) for x in (source / "copies.jsonl").read_text().splitlines()]
    selected = {p.id for p, _ in prepared}
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


def test_pilot_rejects_mls_submission():
    from experiments.geostate.run_e0_jobs import main

    with pytest.raises(SystemExit) as error:
        main(["--config", "e0-synthetic-figure-short50", "--dry"])
    assert error.value.code == 2
