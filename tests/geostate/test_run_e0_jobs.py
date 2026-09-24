"""The MLS launcher: manifest shape, path rewriting, dry run — no submission."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.geostate import run_e0_jobs as L
from experiments.geostate.configs import ARTIFACT_ROOT, TLS_BUNDLE_NAME, config_named


def test_payload_runs_the_config_from_the_staged_commit():
    cfg = config_named("e0-synthetic-text")
    out = L.artifact_dir(cfg, None)
    assert str(out).startswith(ARTIFACT_ROOT) and out.name == cfg.name
    plan = L.payload(
        "abc123", out, "/mnt/virtual_ai0001053-00054_SR004-nfs2/x/.env", cfg
    )
    job = plan["job"]
    assert job["description"] == "[gigaevo/geostate] e0-synthetic-text"
    env = job["environment"]["variables"]
    assert env["PYTHONPATH"].endswith("/code/abc123")
    assert env["AIRI_ENV_FILE"].startswith("/workspace-SR004.nfs2/")
    assert env["SSL_CERT_FILE"].startswith("/workspace-SR004.nfs2/") and env[
        "SSL_CERT_FILE"
    ].endswith(TLS_BUNDLE_NAME)
    assert job["resource"]["instance_type"] == cfg.instance_type
    script = job["script"]
    assert "experiments.geostate.consistency" in script
    assert "--config e0-synthetic-text" in script and "--run" in script
    assert "--summary" in script and "/workspace-SR004.nfs2/" in script
    assert "--limit" not in script
    pilot = L.payload("abc123", L.artifact_dir(cfg, 12), "/tmp/.env", cfg, limit=12)
    assert "--limit 12" in pilot["job"]["script"]
    assert pilot["job"]["description"].endswith("limit12")
    assert L.artifact_dir(cfg, 12).name == "e0-synthetic-text-limit12"


def test_dry_run_prints_the_manifest_and_submits_nothing(monkeypatch, capsys):
    class Creds:
        source = "/mnt/virtual_ai0001053-00054_SR004-nfs2/d/.env"

    monkeypatch.setattr(L, "resolve", lambda: Creds())
    monkeypatch.setattr(L, "git", lambda *a: "deadbeef")
    monkeypatch.setattr(
        L, "mls", lambda *a: (_ for _ in ()).throw(AssertionError("mls"))
    )
    assert L.main(["--config", "e0-aux-figure", "--dry", "--limit", "3"]) == 0
    text = capsys.readouterr().out
    plan = json.loads(text[: text.rindex("}") + 1])
    assert plan["job"]["environment"]["variables"]["PYTHONPATH"].endswith("deadbeef")
    assert "Would freeze" in text


def test_parse_mls_output_takes_the_trailing_json():
    out = "{'diag': 1}\n" + json.dumps({"job_name": "lm-mpi-job-x"}) + "\n"
    assert L.parse_mls_output(out) == {"job_name": "lm-mpi-job-x"}
    assert L.normalized("[gigaevo/geostate] e0 #tag @user") == "[gigaevo/geostate] e0"
    assert Path(L.MLS).name == "mls"
