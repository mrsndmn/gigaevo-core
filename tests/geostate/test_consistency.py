"""E0 copies, evaluation, agreement and the offline summary — no network."""

from __future__ import annotations

import json
import random

import pytest

from experiments.geostate import consistency as C
from experiments.geostate import dsl
from experiments.geostate.configs import E0_GATE, TOL, E0Config, config_named
from experiments.geostate.generator import generate

CFG = config_named("e0-synthetic-text")


def _golden(copy: C.Copy) -> str:
    p = copy.problem
    return (
        "```construction\n"
        + dsl.trajectory_program(p.trajectory, p.answer, p.state())
        + "\n```"
    )


def test_copies_cover_every_kind_and_are_exact_problems():
    problem = generate(seed=3, per_bucket=2)[0]
    copies = C.make_copies(problem, CFG, random.Random(0))
    assert len(copies) == CFG.copies
    assert copies[0].kind == "identity" and copies[0].problem is problem
    assert {c.kind for c in copies[1:]} == set(CFG.kinds) - {"identity"}
    for c in copies[1:]:
        for x, y in c.problem.given.values():
            assert x == round(x, 2) and y == round(y, 2)
        assert c.problem.n_moves == problem.n_moves
        assert c.problem.target_kind == problem.target_kind
        assert sorted(c.problem.given) == sorted(c.perm.values())
        if "relabel" in c.kind:
            assert c.perm != {label: label for label in problem.given}
            assert (
                c.problem.description != problem.description or len(problem.given) < 2
            )
        if c.kind in ("translate", "rotate", "scale", "similarity"):
            assert c.perm == {label: label for label in problem.given}
        if c.kind == "translate":
            assert c.g.theta == 0.0 and c.g.scale == 1.0
        if c.kind == "scale":
            assert c.g.tx == 0.0 and c.g.ty == 0.0 and c.g.theta == 0.0
        assert not c.g.reflect
        d = c.to_dict()
        assert (
            C.Copy.from_dict(json.loads(json.dumps(d))).problem.to_dict()
            == c.problem.to_dict()
        )


def test_golden_programs_succeed_and_agree_across_copies():
    for problem in generate(seed=4, per_bucket=2):
        copies = C.make_copies(problem, CFG, random.Random(1))
        records = [C.evaluate(c, _golden(c)) for c in copies]
        assert all(r["success"] for r in records)
        assert (
            C.pairwise_agreement(records, problem.state().scale, TOL.agreement) == 1.0
        )


def test_original_program_fails_on_a_relabelled_copy():
    problem = generate(seed=4, per_bucket=2)[0]
    copies = C.make_copies(problem, CFG, random.Random(2))
    relabelled = next(c for c in copies if c.kind == "relabel")
    original = (
        "```construction\n"
        + dsl.trajectory_program(problem.trajectory, problem.answer, problem.state())
        + "\n```"
    )
    record = C.evaluate(relabelled, original)
    assert not record["success"]


def test_agreement_semantics():
    problem = generate(seed=5, per_bucket=1)[0]
    scale = problem.state().scale
    copies = C.make_copies(problem, CFG, random.Random(3))
    identity, moved = copies[0], next(c for c in copies if c.kind == "similarity")
    wrong = "```construction\nl = line({a}, {b})\nanswer l\n```"
    labels = list(identity.problem.given)
    a = C.evaluate(identity, wrong.format(a=labels[0], b=labels[1]))
    b = C.evaluate(
        moved, wrong.format(a=moved.perm[labels[0]], b=moved.perm[labels[1]])
    )
    assert not a["success"] and not b["success"]
    assert C.agree(
        a, b, scale, TOL.agreement
    )  # same wrong object in the original frame
    other = C.evaluate(
        moved, wrong.format(a=moved.perm[labels[1]], b=moved.perm[labels[2]])
    )
    assert not C.agree(a, other, scale, TOL.agreement)
    err = C.evaluate(identity, "garbage")
    assert err["error_kind"] == dsl.ERR_SYNTAX
    assert C.agree(err, C.evaluate(moved, "garbage"), scale, TOL.agreement)
    assert not C.agree(err, a, scale, TOL.agreement)
    none = C.evaluate(identity, None)
    assert none["error_kind"] == "request_error" and not none["success"]


def test_prepare_is_idempotent_and_summary_reads_records(tmp_path, monkeypatch):
    cfg = E0Config(
        name="e0-test",
        models=("fake/model",),
        arm="text",
        problem_set="synthetic",
        per_bucket=2,
        copies=4,
        bootstrap=20,
    )
    out = tmp_path / "e0"
    prepared = C.prepare(cfg, out)
    again = C.prepare(cfg, out)
    assert [p.to_dict() for p, _ in prepared] == [p.to_dict() for p, _ in again]
    assert (out / "problems.jsonl").exists() and (out / "copies.jsonl").exists()
    assert len(prepared) == 6 and all(len(c) == 4 for _, c in prepared)

    # fabricate evaluations: a model that solves everything up to 4 moves and
    # writes garbage beyond that
    for problem, copies in prepared:
        for copy in copies:
            content = _golden(copy) if problem.n_moves <= 4 else "no idea"
            record = C.evaluate(copy, content)
            record.update(
                {
                    "model": "fake/model",
                    "problem_id": problem.id,
                    "copy": copy.id,
                    "kind": copy.kind,
                    "bucket": problem.bucket,
                }
            )
            C.write_json(
                C.job_dir(out, "fake/model", problem, copy) / "eval.json", record
            )
    payload, text = C.summarize(cfg, out)
    assert "Kill gate" in text
    rows = {(r["bucket"]): r for r in payload["summary"]}
    assert rows["1-3"]["success_all"] == 1.0 and rows["1-3"]["consistency"] == 1.0
    assert rows["10+"]["success_all"] == 0.0
    assert rows["10+"]["consistency"] == 1.0  # consistently wrong is still consistent
    assert f">={E0_GATE.min_moves} moves" in rows
    assert payload["gate"]["fake/model"] == pytest.approx(1.0)
    assert "H0 FALSIFIED" in text

    # a --dry / --summary invocation through the CLI touches no network
    monkeypatch.setattr(
        C, "make_client", lambda cfg: (_ for _ in ()).throw(AssertionError("network"))
    )
    monkeypatch.setattr(C, "config_named", lambda name: cfg)
    assert C.main(["--config", "e0-test", "--dry", "--out", str(out)]) == 0
    assert C.main(["--config", "e0-test", "--summary", "--out", str(out)]) == 0
    assert (out / "summary.md").exists()


def test_figure_arm_renders_the_given_points(tmp_path):
    problem = generate(seed=6, per_bucket=1)[0]
    copy = C.make_copies(problem, CFG, random.Random(0))[0]
    messages = C.messages_for(copy, "figure", tmp_path)
    assert (tmp_path / "figure.png").exists()
    content = messages[1]["content"]
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "read their positions from the figure" in content[0]["text"]
    for label, (x, y) in problem.given.items():
        assert f"{label} = (" not in content[0]["text"]
    text_messages = C.messages_for(copy, "text", tmp_path)
    assert "Given points:" in text_messages[1]["content"]
