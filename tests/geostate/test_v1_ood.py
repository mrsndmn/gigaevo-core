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
        reflect = next(c for c in CFG.conditions if c.name == name).reflect
        for p in posed:
            original = by_id[p.id]  # ids are the originals: episodes pair by id
            assert p.meta["condition"] == name
            assert p.n_moves == original.n_moves and p.bucket == original.bucket
            if not reflect:  # the same construction, not an easier/harder one
                assert [m.to_dict() for m in p.trajectory] == [
                    m.to_dict() for m in original.trajectory
                ]
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
    path = out / "tokens-s1" / "identity" / "eval.jsonl"
    lines = path.read_text().splitlines()
    assert len(lines) == result["conditions"]["identity"]["n"]
    # a job killed mid-write leaves a torn last line: the rerun re-rolls that
    # episode on a fresh line instead of gluing it onto the fragment
    path.write_text("\n".join(lines[:-1]) + "\n" + lines[-1][:20])
    v1_ood.evaluate_run(cfg, ARM_TOKENS, 1, out, root, model=model)
    episodes = v1_ood._read_episodes(path)
    assert len({e["problem_id"] for e in episodes}) == len(lines)
    with pytest.raises(ValueError, match="arm"):
        v1_ood.evaluate_run(cfg, ARM_TABLE, 1, out, root, model=model)


def _fake(out, cfg, success, n=40, incomplete=(), source=None):
    """success[(cond, arm)] -> rate on long problems; short ones always succeed.

    With `source`, the identity episodes are also written as the saved V1
    episodes of each arm x seed, so identity reproduces them exactly.
    """
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
                if source is not None and c.name == "identity":
                    saved = v1_ood.source_dir(cfg, source, arm, seed)
                    saved.mkdir(parents=True, exist_ok=True)
                    (saved / "eval.jsonl").write_text((d / "eval.jsonl").read_text())


def test_report_gates_on_robustness_not_on_the_absolute_gap(tmp_path, monkeypatch):
    monkeypatch.setattr(v1_ood, "H4_GATE", type(v1_ood.H4_GATE)(n_min=10))
    arms = (ARM_TOKENS, ARM_TABLE)
    src = tmp_path / "v1"

    def run(name, success, **kw):
        _fake(tmp_path / name, CFG, success, source=src / name, **kw)
        return v1_ood.report(CFG, tmp_path / name, src / name, seeds=(1, 2))

    # H1's situation: the table leads at identity; both arms equally robust
    gap = {
        (c.name, a): 0.9 if a == ARM_TABLE else 0.7
        for c in CFG.conditions
        for a in arms
    }
    payload, text = run("gap", gap)
    assert payload["gate"]["far"] == ["translate-far", "scale-up"]
    assert payload["deltas"]["identity"]["delta_pts"] < -15  # absolute gap reported
    assert payload["robustness"]["scale-up"]["delta_pts"] == pytest.approx(0.0)
    assert payload["gate"]["verdict"] == "KILL"
    # the table drops to the tokens' level on a far pose, the tokens do not
    # move: the absolute gap closes to zero, robustness is +20 -> GO
    drop = dict(gap)
    drop[("translate-far", ARM_TABLE)] = 0.7
    payload, text = run("drop", drop)
    assert payload["deltas"]["translate-far"]["delta_pts"] == pytest.approx(0.0)
    assert payload["robustness"]["translate-far"]["delta_pts"] > 10
    assert payload["robustness"]["translate-far"]["ci_pts"][0] > 0
    assert payload["gate"]["verdict"] == "GO" and "translate-far" in text
    assert payload["rows"]["translate-far/table"]["vs_identity"]["delta_pts"] < -15
    # a drop on a reported-only condition (scale-down: print precision) or on
    # reflect does not count
    near = dict(gap)
    near[("scale-down", ARM_TABLE)] = 0.2
    near[("reflect", ARM_TABLE)] = 0.2
    payload, _ = run("near", near)
    assert payload["robustness"]["scale-down"]["delta_pts"] > 10
    assert payload["gate"]["verdict"] == "KILL"
    # tokens more robust, but between the margins -> RECONSIDER (the fake's
    # rates step by 10 pts, so the go margin is raised to leave room)
    mid = dict(gap)
    mid[("scale-up", ARM_TABLE)] = 0.8
    monkeypatch.setattr(
        v1_ood, "H4_GATE", type(v1_ood.H4_GATE)(n_min=10, go_delta_pts=15.0)
    )
    payload, _ = run("mid", mid)
    assert payload["robustness"]["scale-up"]["delta_pts"] == pytest.approx(10.0)
    assert payload["gate"]["verdict"] == "RECONSIDER"
    monkeypatch.setattr(v1_ood, "H4_GATE", type(v1_ood.H4_GATE)(n_min=10))
    # an incomplete run withholds the verdict but keeps the numbers
    payload, text = run("part", drop, incomplete={(ARM_TABLE, 2, "rotate")})
    assert payload["gate"]["verdict"] == "withheld" and "table/s2/rotate" in text
    assert payload["robustness"]["translate-far"]["delta_pts"] > 10
    # too few paired long problems withholds too
    monkeypatch.setattr(v1_ood, "H4_GATE", type(v1_ood.H4_GATE)(n_min=1000))
    payload, text = v1_ood.report(CFG, tmp_path / "drop", src / "drop", seeds=(1, 2))
    assert payload["gate"]["verdict"] == "withheld" and "fewer than 1000" in text


def test_identity_must_reproduce_the_saved_v1_episodes(tmp_path, monkeypatch):
    monkeypatch.setattr(v1_ood, "H4_GATE", type(v1_ood.H4_GATE)(n_min=10))
    arms = (ARM_TOKENS, ARM_TABLE)
    root = tmp_path / "v1"
    _fake(
        tmp_path / "out",
        CFG,
        {(c.name, a): 0.8 for c in CFG.conditions for a in arms},
        source=root,
    )
    payload, _ = v1_ood.report(CFG, tmp_path / "out", root, seeds=(1, 2))
    assert payload["reproduction"]["tokens/s1"]["same_moves"] == 1.0
    assert payload["gate"]["verdict"] == "KILL"
    # one saved episode took other moves (same outcome): reported, not fatal
    saved_path = v1_ood.source_dir(CFG, root, ARM_TOKENS, 1) / "eval.jsonl"
    episodes = [json.loads(line) for line in saved_path.read_text().splitlines()]
    episodes[1]["moves"] = ["line(A, B)"]
    saved_path.write_text("".join(json.dumps(e) + "\n" for e in episodes))
    payload, _ = v1_ood.report(CFG, tmp_path / "out", root, seeds=(1, 2))
    r = payload["reproduction"]["tokens/s1"]
    assert r["n"] == 40 and r["same_moves"] == pytest.approx(39 / 40)
    assert r["same_success"] == 1.0 and payload["gate"]["verdict"] == "KILL"
    # 3 of 40 outcomes differ (< 0.95 agree): these are not H1's adapters
    for e in episodes[:3]:
        e["success"] = not e["success"]
    saved_path.write_text("".join(json.dumps(e) + "\n" for e in episodes))
    payload, text = v1_ood.report(CFG, tmp_path / "out", root, seeds=(1, 2))
    assert payload["gate"]["verdict"] == "withheld"
    assert payload["gate"]["unreproduced"] == ["tokens/s1"] and "reproduce" in text
    # no saved episodes at all (or no source root) withholds as well
    payload, _ = v1_ood.report(CFG, tmp_path / "out", None, seeds=(1, 2))
    assert payload["gate"]["verdict"] == "withheld"


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
