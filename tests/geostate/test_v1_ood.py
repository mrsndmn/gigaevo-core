"""H4: pose conditions over the frozen V1 eval set, adapter re-evaluation, gate."""

from __future__ import annotations

from dataclasses import replace
import json
import math

import pytest

from experiments.geostate import v1_ood
from experiments.geostate.configs import (
    ARM_TABLE,
    ARM_TOKENS,
    POSE_CONDITIONS,
    V1_OOD,
    PoseCondition,
)
from experiments.geostate.generator import generate, load_problems, save_problems

CFG = replace(
    V1_OOD, source_protocol="abcdef123456", data_key="test", smoke_per_bucket=1
)


def _source(tmp_path, per_bucket=2):
    root = tmp_path / "v1"
    problems = generate(seed=7, per_bucket=per_bucket)
    save_problems(problems, root / "data-test" / "eval.jsonl")
    return root, problems


def _centroid(given):
    xs = [x for x, _ in given.values()]
    ys = [y for _, y in given.values()]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def test_conditions_keep_ids_and_apply_their_pose(tmp_path):
    root, problems = _source(tmp_path)
    info = v1_ood.prepare(CFG, tmp_path / "out", root)
    assert info["n_source"] == len(problems)
    by_id = {p.id: p for p in problems}
    loaded = {
        c.name: load_problems(tmp_path / "out" / "conditions" / f"{c.name}.jsonl")
        for c in CFG.conditions
    }
    for name, posed in loaded.items():
        counts = info["conditions"][name]
        assert counts["kept"] == len(posed)
        assert counts["kept"] + counts["rejected"] == len(problems)
        for p in posed:
            original = by_id[p.id]  # ids are the originals: episodes pair by id
            assert p.meta["condition"] == name
            assert p.n_moves == original.n_moves and p.bucket == original.bucket
    # identity rebuilds the frozen problems exactly
    for p in loaded["identity"]:
        a, b = p.to_dict(), by_id[p.id].to_dict()
        assert {k: v for k, v in a.items() if k != "meta"} == {
            k: v for k, v in b.items() if k != "meta"
        }
    for p in loaded["scale-up"]:
        for label, (x, y) in p.given.items():
            ox, oy = by_id[p.id].given[label]
            assert (x, y) == pytest.approx((10 * ox, 10 * oy), abs=0.006)
    for p in loaded["translate-far"]:
        cx, cy = _centroid(p.given)
        ox, oy = _centroid(by_id[p.id].given)
        assert math.hypot(cx - ox, cy - oy) == pytest.approx(100.0, abs=0.02)
    assert all(p.meta["pose"]["reflect"] for p in loaded["reflect"])
    assert not any(p.meta["pose"]["reflect"] for p in loaded["rotate"])


def test_prepare_is_deterministic_idempotent_and_fingerprinted(tmp_path):
    root, _ = _source(tmp_path)
    a = v1_ood.prepare(CFG, tmp_path / "a", root)
    b = v1_ood.prepare(CFG, tmp_path / "b", root)
    for c in CFG.conditions:
        fa = (tmp_path / "a" / "conditions" / f"{c.name}.jsonl").read_text()
        fb = (tmp_path / "b" / "conditions" / f"{c.name}.jsonl").read_text()
        assert fa == fb  # poses depend on (seed, condition, id) only
    assert v1_ood.prepare(CFG, tmp_path / "a", root) == a == b  # reuse, no rebuild
    # a directory prepared from another eval set (or smoke) is refused
    with pytest.raises(RuntimeError, match="prepared for"):
        v1_ood.prepare(CFG, tmp_path / "a", root, smoke=True)
    eval_path = root / "data-test" / "eval.jsonl"
    eval_path.write_text(eval_path.read_text().splitlines()[0] + "\n")
    with pytest.raises(RuntimeError, match="prepared for"):
        v1_ood.prepare(CFG, tmp_path / "a", root)


def test_smoke_takes_a_few_problems_per_bucket(tmp_path):
    root, _ = _source(tmp_path)
    info = v1_ood.prepare(CFG, tmp_path / "s", root, smoke=True)
    assert info["n_source"] == 3  # one per bucket


def test_unbuildable_pose_is_rejected_not_kept(tmp_path):
    root, problems = _source(tmp_path)
    # scale 1e-4 puts given points closer than the rounding step: degenerate
    cfg = replace(CFG, conditions=(PoseCondition("tiny", scale=1e-4),))
    info = v1_ood.prepare(cfg, tmp_path / "t", root)
    assert info["conditions"]["tiny"]["rejected"] == len(problems)
    assert info["conditions"]["tiny"]["kept"] == 0


@pytest.mark.timeout(300)
def test_tiny_model_evaluates_every_condition_and_resumes(tmp_path):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(4)
    from tests.geostate.test_v1 import _tiny_model

    root, _ = _source(tmp_path, per_bucket=1)
    cfg = replace(
        CFG,
        conditions=(POSE_CONDITIONS[0], POSE_CONDITIONS[3]),  # identity, scale-up
    )
    model = _tiny_model(ARM_TOKENS)
    adapters = v1_ood.source_dir(cfg, root, ARM_TOKENS, 1) / "adapters.pt"
    model.save_adapters(adapters)
    out = tmp_path / "ood"
    result = v1_ood.evaluate_run(cfg, ARM_TOKENS, 1, out, root, model=model)
    assert result["complete"] and set(result["conditions"]) == {"identity", "scale-up"}
    for name in ("identity", "scale-up"):
        lines = (out / "tokens-s1" / name / "eval.jsonl").read_text().splitlines()
        assert len(lines) == result["conditions"][name]["n"] > 0
    # a second call resumes: no episode is appended twice
    v1_ood.evaluate_run(cfg, ARM_TOKENS, 1, out, root, model=model)
    lines = (out / "tokens-s1" / "identity" / "eval.jsonl").read_text().splitlines()
    assert len(lines) == result["conditions"]["identity"]["n"]
    with pytest.raises(ValueError, match="arm"):
        v1_ood.evaluate_run(cfg, ARM_TABLE, 1, out, root, model=model)


def _fake(out, cfg, success, n=40, incomplete=()):
    """success[(cond, arm)] -> rate on long problems; short ones always succeed."""
    conds = {
        c.name: {"kept": n, "rejected": 0, "rejected_ids": []} for c in cfg.conditions
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "prepare.json").write_text(
        json.dumps({"fingerprint": {"ood": "x"}, "n_source": n, "conditions": conds})
    )
    for c in cfg.conditions:
        for arm in (ARM_TOKENS, ARM_TABLE):
            for seed in (1, 2):
                d = out / f"{arm}-s{seed}" / c.name
                d.mkdir(parents=True)
                rate = success[(c.name, arm)]
                m = n - 5 if (arm, seed, c.name) in incomplete else n
                with (d / "eval.jsonl").open("w") as fh:
                    for i in range(m):
                        long = i % 4 != 0
                        ok = (not long) or ((i * 7 + seed) % 10 < rate * 10)
                        fh.write(
                            json.dumps(
                                {
                                    "problem_id": f"eval-{i}",
                                    "bucket": "4-9" if long else "1-3",
                                    "n_moves_reference": 7 if long else 2,
                                    "success": ok,
                                    "selection_error": False,
                                    "moves": [],
                                }
                            )
                            + "\n"
                        )
                (d / "metrics.json").write_text("{}")


def test_report_reads_the_gate_on_far_conditions(tmp_path, monkeypatch):
    monkeypatch.setattr(v1_ood, "H4_GATE", type(v1_ood.H4_GATE)(n_min=10))
    arms = (ARM_TOKENS, ARM_TABLE)
    flat = {(c.name, a): 0.8 for c in CFG.conditions for a in arms}
    _fake(tmp_path / "flat", CFG, flat)
    payload, text = v1_ood.report(CFG, tmp_path / "flat", seeds=(1, 2))
    assert payload["gate"]["verdict"] == "KILL"
    assert payload["gate"]["far"] == ["translate-far", "scale-up", "scale-down"]
    assert payload["deltas"]["scale-up"]["delta_pts"] == pytest.approx(0.0)
    # the table collapses on one far condition only: tokens win there -> GO
    broken = dict(flat)
    broken[("scale-up", ARM_TABLE)] = 0.2
    _fake(tmp_path / "go", CFG, broken)
    payload, text = v1_ood.report(CFG, tmp_path / "go", seeds=(1, 2))
    assert payload["gate"]["verdict"] == "GO" and "scale-up" in text
    assert payload["rows"]["scale-up/table"]["vs_identity"]["delta_pts"] < -40
    assert payload["rows"]["scale-up/tokens"]["vs_identity"]["delta_pts"] == 0
    # a gain on a non-far condition does not count
    near = dict(flat)
    near[("reflect", ARM_TABLE)] = 0.2
    _fake(tmp_path / "near", CFG, near)
    payload, _ = v1_ood.report(CFG, tmp_path / "near", seeds=(1, 2))
    assert payload["gate"]["verdict"] == "KILL"
    # an incomplete run withholds the verdict but keeps the numbers
    _fake(tmp_path / "part", CFG, broken, incomplete={(ARM_TABLE, 2, "rotate")})
    payload, text = v1_ood.report(CFG, tmp_path / "part", seeds=(1, 2))
    assert payload["gate"]["verdict"] == "withheld" and "table/s2/rotate" in text
    assert payload["deltas"]["scale-up"]["delta_pts"] > 10
    # too few paired long problems withholds too
    monkeypatch.setattr(v1_ood, "H4_GATE", type(v1_ood.H4_GATE)(n_min=1000))
    payload, text = v1_ood.report(CFG, tmp_path / "go", seeds=(1, 2))
    assert payload["gate"]["verdict"] == "withheld" and "fewer than 1000" in text


def test_report_compares_identity_with_the_saved_v1_episodes(tmp_path):
    arms = (ARM_TOKENS, ARM_TABLE)
    _fake(
        tmp_path / "out", CFG, {(c.name, a): 0.8 for c in CFG.conditions for a in arms}
    )
    root = tmp_path / "v1"
    saved = v1_ood.source_dir(CFG, root, ARM_TOKENS, 1)
    saved.mkdir(parents=True)
    fresh = (tmp_path / "out" / "tokens-s1" / "identity" / "eval.jsonl").read_text()
    episodes = [json.loads(line) for line in fresh.splitlines()]
    episodes[1]["moves"] = ["line(A, B)"]
    (saved / "eval.jsonl").write_text("".join(json.dumps(e) + "\n" for e in episodes))
    payload, text = v1_ood.report(CFG, tmp_path / "out", root, seeds=(1, 2))
    r = payload["reproduction"]["tokens/s1"]
    assert r["n"] == 40 and r["same_moves"] == pytest.approx(39 / 40)
    assert r["same_success"] == 1.0 and payload["reproduction"]["table/s1"] is None


def test_ood_launcher_payload(tmp_path):
    from experiments.geostate import run_v1_ood_jobs as L

    out = v1_ood.default_out(V1_OOD)
    plan = L.payload("abc", out, v1_ood.default_source_root(), V1_OOD, ARM_TABLE, 2)
    job = plan["job"]
    assert job["description"] == (
        f"[gigaevo/geostate] v1-ood/{v1_ood.ood_hash(V1_OOD)} table s2"
    )
    assert out.name == f"v1-ood-{v1_ood.ood_hash(V1_OOD)}"
    script = job["script"]
    assert "experiments.geostate.v1_ood --eval --arm table --seed 2" in script
    assert "--train" not in script
    assert script.count("/workspace-SR004.nfs2/") >= 3  # workdir, out, source root
    assert "/mnt/" not in script
    env = job["environment"]["variables"]
    assert env["HF_HUB_OFFLINE"] == "1" and env["HF_HOME"].startswith("/workspace")
