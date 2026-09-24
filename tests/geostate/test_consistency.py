"""E0 copies, evaluation, agreement, recovery and the offline summary — no network."""

from __future__ import annotations

import json
import random

import pytest

from experiments.geostate import consistency as C
from experiments.geostate import dsl
from experiments.geostate.configs import (
    E0_GATE,
    GROUP_OOD_REFLECTION,
    OOD_KINDS,
    TOL,
    E0Config,
    config_named,
)
from experiments.geostate.generator import generate

CFG = config_named("e0-synthetic-text")


def _golden(copy: C.Copy) -> str:
    p = copy.problem
    program = dsl.trajectory_program(p.trajectory, p.answer, p.state())
    return "```construction\n" + program + "\n```"


def _wrong(copy: C.Copy, i: int, j: int) -> str:
    labels = list(copy.problem.given)
    return f"```construction\nl = line({labels[i]}, {labels[j]})\nanswer l\n```"


def test_copies_cover_every_kind_and_are_exact_problems():
    problem = generate(seed=3, per_bucket=2)[0]
    copies = C.make_copies(problem, CFG, random.Random(0))
    assert len(copies) == CFG.copies
    assert copies[0].kind == "identity" and copies[0].problem is problem
    assert {c.kind for c in copies[1:]} <= set(CFG.kinds) - {"identity"}
    keys = set()
    for c in copies:
        key = (
            json.dumps(c.problem.to_dict()["given"], sort_keys=True)
            + c.problem.description
        )
        assert key not in keys  # no two copies coincide
        keys.add(key)
    for c in copies[1:]:
        for x, y in c.problem.given.values():
            assert x == round(x, 2) and y == round(y, 2)
        assert c.problem.n_moves == problem.n_moves
        assert c.problem.target_kind == problem.target_kind
        assert sorted(c.problem.given) == sorted(c.perm.values())
        if "relabel" in c.kind:
            assert c.perm != {label: label for label in problem.given}
        else:
            assert c.perm == {label: label for label in problem.given}
        if c.kind == "translate":
            assert c.g.theta == 0.0 and c.g.scale == 1.0
        if c.kind == "scale":
            assert c.g.tx == 0.0 and c.g.ty == 0.0 and c.g.theta == 0.0
        assert not c.g.reflect
        d = c.to_dict()
        back = C.Copy.from_dict(json.loads(json.dumps(d)))
        assert back.problem.to_dict() == c.problem.to_dict()


def test_two_point_problems_get_distinct_copies():
    problem = next(p for p in generate(seed=12, per_bucket=6) if len(p.given) >= 3)
    two = problem  # relabel of a 3+ point problem always has alternatives
    copies = C.make_copies(two, CFG, random.Random(5))
    assert len({c.problem.description + str(c.problem.given) for c in copies}) == len(
        copies
    )


def test_reflection_copies_keep_only_well_posed_targets():
    cfg = E0Config(
        name="ood",
        models=("m",),
        arm="text",
        problem_set="synthetic",
        copies=4,
        kinds=OOD_KINDS,
        group=GROUP_OOD_REFLECTION,
    )
    rng = random.Random(1)
    seen_reflect = False
    for problem in generate(seed=13, per_bucket=4):
        copies = C.make_copies(problem, cfg, rng)
        for c in copies[1:]:
            assert c.g.reflect and c.kind in ("reflect", "reflect+relabel")
            seen_reflect = True
            # the copy's target is the reflection of the original target
            back = C.inverse_params(c.problem.target_kind, c.problem.target_params, c.g)
            scale = problem.state().scale
            assert (
                C.distance_between(
                    problem.target_kind, back, problem.target_params, scale
                )
                <= TOL.agreement * scale
            )
        if problem.expr[0] in ("rotate90", "rotate270"):
            assert len(copies) == 1  # orientation-dependent: no reflected copy survives
    assert seen_reflect


def test_golden_programs_succeed_and_agree_across_copies():
    for problem in generate(seed=4, per_bucket=2):
        copies = C.make_copies(problem, CFG, random.Random(1))
        records = [C.evaluate(c, _golden(c)) for c in copies]
        assert all(r["success"] and r["answered"] for r in records)
        agreement, n_pairs = C.pairwise_agreement(
            records, problem.state().scale, TOL.agreement
        )
        assert agreement == 1.0 and n_pairs == len(copies) * (len(copies) - 1) // 2


def test_original_program_fails_on_a_relabelled_copy():
    problem = generate(seed=4, per_bucket=2)[0]
    copies = C.make_copies(problem, CFG, random.Random(2))
    relabelled = next(c for c in copies if c.kind == "relabel")
    original = (
        "```construction\n"
        + dsl.trajectory_program(problem.trajectory, problem.answer, problem.state())
        + "\n```"
    )
    assert not C.evaluate(relabelled, original)["success"]


def test_agreement_semantics():
    problem = next(p for p in generate(seed=5, per_bucket=3) if len(p.given) >= 3)
    scale = problem.state().scale
    copies = C.make_copies(problem, CFG, random.Random(3))
    identity = copies[0]
    moved = next(c for c in copies if c.kind == "similarity")
    labels = list(identity.problem.given)
    a = C.evaluate(identity, _wrong(identity, 0, 1))
    same = f"```construction\nl = line({moved.perm[labels[0]]}, {moved.perm[labels[1]]})\nanswer l\n```"
    b = C.evaluate(moved, same)
    assert not a["success"] and not b["success"] and a["answered"] and b["answered"]
    assert (
        C.agree(a, b, scale, TOL.agreement) is True
    )  # same wrong line, original frame
    other = f"```construction\nl = line({moved.perm[labels[1]]}, {moved.perm[labels[2]]})\nanswer l\n```"
    assert C.agree(a, C.evaluate(moved, other), scale, TOL.agreement) is False
    # a wrong line 45 degrees off is not the same line
    # not answered / request error: excluded, never "agreement"
    err = C.evaluate(identity, "garbage")
    assert err["error_kind"] == dsl.ERR_SYNTAX and not err["answered"]
    assert C.agree(err, C.evaluate(moved, "garbage"), scale, TOL.agreement) is None
    assert C.agree(err, a, scale, TOL.agreement) is None
    none = C.evaluate(identity, None)
    assert none["error_kind"] == C.REQUEST_ERROR and not none["answered"]
    assert C.agree(none, a, scale, TOL.agreement) is None
    assert C.pairwise_agreement([err, none], scale, TOL.agreement) == (None, 0)
    # success vs wrong answer: disagreement
    good = C.evaluate(identity, _golden(identity))
    assert C.agree(good, a, scale, TOL.agreement) is False


def test_line_agreement_bounds_the_angle():
    from experiments.geostate.macros import distance_between

    a = (1.0, 0.0, 2.0)
    b45 = (2**-0.5, 2**-0.5, 2.0)
    assert distance_between("line", a, b45, scale=20.0) > TOL.agreement * 20.0
    b_small = (0.9999, 0.01, 2.0)
    assert distance_between("line", a, b_small, scale=20.0) <= TOL.agreement * 20.0
    assert distance_between("line", a, (-1.0, 0.0, -2.0), scale=20.0) == 0.0


def _fake_run(cfg: E0Config, out, solve_up_to: int, errors: set[str] = frozenset()):
    """Write eval.json files as `run_one` would, without any network."""
    prepared = C.prepare(cfg, out)
    for problem, copies in prepared:
        for copy in copies:
            directory = C.job_dir(out, cfg.models[0], problem, copy)
            if f"{problem.id}/{copy.id}" in errors:
                C.write_json(
                    directory / "result.json",
                    {"error": "AuthenticationError: expired", "content": None},
                )
                continue
            content = _golden(copy) if problem.n_moves <= solve_up_to else "no idea"
            record = C.evaluate(copy, content)
            record.update(
                {
                    "model": cfg.models[0],
                    "problem_id": problem.id,
                    "copy": copy.id,
                    "kind": copy.kind,
                    "bucket": problem.bucket,
                }
            )
            C.write_json(directory / "eval.json", record)
    return prepared


def _cfg(name="e0-test", **kw) -> E0Config:
    base = dict(
        name=name,
        models=("fake/model",),
        arm="text",
        problem_set="synthetic",
        per_bucket=2,
        copies=4,
        bootstrap=20,
        gated=True,
    )
    base.update(kw)
    return E0Config(**base)


def test_prepare_is_idempotent_and_checks_the_fingerprint(tmp_path):
    cfg = _cfg()
    out = tmp_path / "e0"
    prepared = C.prepare(cfg, out)
    again = C.prepare(cfg, out)
    assert [p.to_dict() for p, _ in prepared] == [p.to_dict() for p, _ in again]
    assert len(prepared) == 6 and all(len(c) == 4 for _, c in prepared)
    manifest = C.read_json(out / "prepare.json")
    assert manifest["fingerprint"] == C.fingerprint(cfg)
    assert manifest["generator"]["accepted"] == 6
    with pytest.raises(RuntimeError, match="different config"):
        C.prepare(_cfg(copies=5), out)


def test_summary_withholds_the_verdict_without_answers(tmp_path):
    cfg = _cfg()
    out = tmp_path / "e0"
    _fake_run(cfg, out, solve_up_to=4)
    payload, text = C.summarize(cfg, out)
    rows = {r["bucket"]: r for r in payload["summary"]}
    assert rows["1-3"]["success_identity"] == 1.0 and rows["1-3"]["consistency"] == 1.0
    assert rows["10+"]["answered"] == 0.0
    assert rows["10+"]["n_consistency"] == 0  # unanswered copies are not "consistent"
    gate_row = rows[f">={E0_GATE.min_moves} moves"]
    assert gate_row["coverage"] == 1.0
    status = payload["gate"]["models"]["fake/model"]
    assert not status["applicable"] and any("answered" in r for r in status["withheld"])
    assert payload["gate"]["verdict"] == "withheld"
    assert "H0 FALSIFIED" not in text and "withheld" in text


def test_summary_reports_missing_requests_and_never_counts_them(tmp_path):
    cfg = _cfg()
    out = tmp_path / "e0"
    prepared = C.prepare(cfg, out)
    long_problem = next(p for p, _ in prepared if p.n_moves >= E0_GATE.min_moves)
    _fake_run(cfg, out, solve_up_to=99, errors={f"{long_problem.id}/c1"})
    payload, text = C.summarize(cfg, out)
    gate_row = next(r for r in payload["summary"] if r["bucket"].startswith(">="))
    assert gate_row["n_missing"] == 1 and gate_row["coverage"] < 1.0
    assert payload["gate"]["verdict"] == "withheld"
    assert "not evaluated" in text
    # recovery archives exactly the failed request; a *held* lock blocks it,
    # a stale lock file left by a dead run does not
    with C.RunLock(out):
        assert C.lock_is_held(out)
        with pytest.raises(RuntimeError, match="live run"):
            C.recover_errors(out)
        with pytest.raises(RuntimeError, match="another run"):
            with C.RunLock(out):
                pass
    assert (out / C.LOCK_NAME).exists() and not C.lock_is_held(out)
    archived = C.recover_errors(out)
    assert len(archived) == 1 and archived[0].name == "c1.failed1"
    assert not (archived[0].parent / "c1").exists()
    # with everything answered and solved, the gate can be read and trips
    _fake_run(cfg, out, solve_up_to=99)
    payload, text = C.summarize(cfg, out)
    assert payload["gate"]["verdict"] == "H0 FALSIFIED"
    assert payload["gate"]["models"]["fake/model"]["applicable"]


def test_consistently_wrong_but_solving_model_can_trip_the_gate(tmp_path):
    """A model that solves the identity copies and answers everything is judged."""
    cfg = _cfg()
    out = tmp_path / "e0"
    _fake_run(cfg, out, solve_up_to=99)
    payload, _ = C.summarize(cfg, out)
    gate_row = next(r for r in payload["summary"] if r["bucket"].startswith(">="))
    assert gate_row["success_identity"] == 1.0 and gate_row["answered"] == 1.0
    assert payload["gate"]["verdict"] == "H0 FALSIFIED"


def test_ungated_config_prints_no_verdict(tmp_path):
    cfg = _cfg(name="e0-figure-like", gated=False)
    out = tmp_path / "e0"
    _fake_run(cfg, out, solve_up_to=99)
    payload, text = C.summarize(cfg, out)
    assert payload["gate"]["verdict"] is None
    assert "Not a gated config" in text and "FALSIFIED" not in text


def test_by_kind_keeps_every_copy_and_delta_success_is_reported(tmp_path):
    cfg = _cfg(copies=8)
    out = tmp_path / "e0"
    _fake_run(cfg, out, solve_up_to=99)
    payload, text = C.summarize(cfg, out)
    kinds = {}
    for r in payload["rows"]:
        for kind, values in r["by_kind"].items():
            kinds[kind] = kinds.get(kind, 0) + len(values)
    assert sum(kinds.values()) == 6 * 8
    assert kinds["relabel"] == 12  # two relabel copies per problem, both kept
    all_row = next(r for r in payload["summary"] if r["bucket"] == "all")
    assert all_row["delta_success"] == 0.0 and all_row["delta_success_ci"] == [0.0, 0.0]
    assert "Δsuccess" in text


def test_stratified_limit_spans_buckets(tmp_path):
    cfg = _cfg(per_bucket=3)
    prepared = C.prepare(cfg, tmp_path / "e0")
    taken = C.stratified(prepared, 4)
    assert [p.bucket for p, _ in taken] == ["1-3", "4-9", "10+", "1-3"]
    assert C.stratified(prepared, None) is prepared


def test_cli_dry_and_summary_touch_no_network(tmp_path, monkeypatch):
    cfg = _cfg()
    out = tmp_path / "e0"
    monkeypatch.setattr(
        C, "make_client", lambda cfg: (_ for _ in ()).throw(AssertionError("network"))
    )
    monkeypatch.setattr(C, "config_named", lambda name: cfg)
    assert C.main(["--config", "e0-test", "--dry", "--out", str(out)]) == 0
    assert C.main(["--config", "e0-test", "--summary", "--out", str(out)]) == 0
    assert (out / "summary.md").exists() and (out / "summary.json").exists()
    assert C.main(["--config", "e0-test", "--recover-errors", "--out", str(out)]) == 0


def test_figure_arm_renders_the_given_points(tmp_path):
    problem = generate(seed=6, per_bucket=1)[0]
    copy = C.make_copies(problem, CFG, random.Random(0))[0]
    messages = C.messages_for(copy, "figure", tmp_path)
    assert (tmp_path / "figure.png").exists()
    content = messages[1]["content"]
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "read their positions from the figure" in content[0]["text"]
    assert not any(ch.isdigit() for ch in content[0]["text"].replace("90-degree", ""))
    text_messages = C.messages_for(copy, "text", tmp_path)
    assert "Given points:" in text_messages[1]["content"]


def test_gate_is_withheld_when_any_model_lacks_data(tmp_path):
    """A model with no evaluations, or a missing copy, blocks 'H0 stands'."""
    cfg = _cfg(models=("fake/a", "fake/b"))
    out = tmp_path / "e0"
    prepared = C.prepare(cfg, out)
    long_problem = next(p for p, _ in prepared if p.n_moves >= E0_GATE.min_moves)
    # model a: answers everything, solves everything, but inconsistent-ish is
    # irrelevant here — it is below the gate only if we make it disagree, so
    # give it one wrong copy per problem to sit below 0.95
    for problem, copies in prepared:
        for i, copy in enumerate(copies):
            content = _wrong(copy, 0, 1) if i == 1 else _golden(copy)
            record = C.evaluate(copy, content)
            record.update(
                {
                    "model": "fake/a",
                    "problem_id": problem.id,
                    "copy": copy.id,
                    "kind": copy.kind,
                    "bucket": problem.bucket,
                }
            )
            C.write_json(C.job_dir(out, "fake/a", problem, copy) / "eval.json", record)
    payload, text = C.summarize(cfg, out)
    assert payload["gate"]["models"]["fake/a"]["applicable"]
    assert payload["gate"]["models"]["fake/a"]["consistency"] < E0_GATE.consistency_kill
    assert payload["gate"]["models"]["fake/b"]["incomplete"]
    assert (
        payload["gate"]["verdict"] == "withheld"
        and "incomplete data for fake/b" in text
    )
    # model b evaluated except one copy of a long problem: still withheld
    for problem, copies in prepared:
        for copy in copies:
            if problem is long_problem and copy.id == "c1":
                continue
            record = C.evaluate(copy, _golden(copy))
            record.update(
                {
                    "model": "fake/b",
                    "problem_id": problem.id,
                    "copy": copy.id,
                    "kind": copy.kind,
                    "bucket": problem.bucket,
                }
            )
            C.write_json(C.job_dir(out, "fake/b", problem, copy) / "eval.json", record)
    payload, text = C.summarize(cfg, out)
    assert payload["gate"]["verdict"] == "withheld" and "1 copies not evaluated" in text
    # the last copy arrives and model b is perfectly consistent: falsified,
    # regardless of model a
    copy = next(
        c for p, cs in prepared if p is long_problem for c in cs if c.id == "c1"
    )
    record = C.evaluate(copy, _golden(copy))
    record.update(
        {
            "model": "fake/b",
            "problem_id": long_problem.id,
            "copy": copy.id,
            "kind": copy.kind,
            "bucket": long_problem.bucket,
        }
    )
    C.write_json(C.job_dir(out, "fake/b", long_problem, copy) / "eval.json", record)
    payload, text = C.summarize(cfg, out)
    assert payload["gate"]["verdict"] == "H0 FALSIFIED"


def test_gate_names_models_excluded_for_quality(tmp_path):
    cfg = _cfg(models=("fake/a", "fake/b"))
    out = tmp_path / "e0"
    prepared = C.prepare(cfg, out)
    for problem, copies in prepared:
        for i, copy in enumerate(copies):
            a = _wrong(copy, 0, 1) if i == 1 else _golden(copy)  # below the gate
            b = "no idea"  # answers nothing
            for model, content in (("fake/a", a), ("fake/b", b)):
                record = C.evaluate(copy, content)
                record.update(
                    {
                        "model": model,
                        "problem_id": problem.id,
                        "copy": copy.id,
                        "kind": copy.kind,
                        "bucket": problem.bucket,
                    }
                )
                C.write_json(C.job_dir(out, model, problem, copy) / "eval.json", record)
    payload, text = C.summarize(cfg, out)
    assert payload["gate"]["verdict"] == "H0 stands"
    assert "Excluded for answered/success conditions: fake/b" in text


def test_recover_errors_is_idempotent_and_handles_interrupted_requests(tmp_path):
    cfg = _cfg()
    out = tmp_path / "e0"
    prepared = C.prepare(cfg, out)
    problem, copies = prepared[0]
    failed = C.job_dir(out, cfg.models[0], problem, copies[0])
    C.write_json(failed / "result.json", {"error": "boom", "content": None})
    interrupted = C.job_dir(out, cfg.models[0], problem, copies[1])
    C.write_json(interrupted / "request.json", {"model": "x"})
    fine = C.job_dir(out, cfg.models[0], problem, copies[2])
    C.write_json(fine / "result.json", {"error": None, "content": "answer A"})
    untouched = C.job_dir(out, cfg.models[0], problem, copies[3])
    untouched.mkdir(parents=True)
    first = C.recover_errors(out)
    assert sorted(p.name for p in first) == ["c0.failed1", "c1.failed1"]
    assert fine.exists() and untouched.exists()
    assert C.recover_errors(out) == []  # archives are never re-archived
    C.write_json(failed / "result.json", {"error": "again", "content": None})
    assert [p.name for p in C.recover_errors(out)] == ["c0.failed2"]


def test_fingerprint_covers_tolerances(tmp_path):
    from experiments.geostate.configs import ToleranceSpec

    cfg = _cfg()
    out = tmp_path / "e0"
    C.prepare(cfg, out)
    with pytest.raises(RuntimeError, match="different config"):
        C.prepare(cfg, out, tol=ToleranceSpec(agreement=0.05))


def test_client_verifies_tls_with_the_intermediate_bundle(tmp_path, monkeypatch):
    from experiments.geostate import client as CL
    from experiments.geostate.configs import INTERMEDIATE_CERT

    bundle = CL.ca_bundle(tmp_path)
    text = bundle.read_text()
    assert INTERMEDIATE_CERT.read_text().strip() in text
    assert text.count("BEGIN CERTIFICATE") > 10  # certifi roots are in front
    assert CL.ca_bundle(tmp_path) == bundle  # written once
    bundle.write_text("stale")
    assert CL.ca_bundle(tmp_path).read_text() == text  # refreshed when it differs

    captured = {}

    class FakeOpenAI:
        def __init__(self, **kw):
            captured.update(kw)

    class Creds:
        base_url = CFG.base_url
        api_key = "token"
        source = "test"
        expires_at = None

    monkeypatch.setattr(CL._credentials, "resolve", lambda expected_base_url: Creds())
    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    CL.make_client(CFG, tmp_path)
    assert captured["max_retries"] == 0
    context = captured["http_client"]._transport._pool._ssl_context
    subjects = [
        dict(x for rdn in cert["subject"] for x in rdn)
        for cert in context.get_ca_certs()
    ]
    assert any("GlobalSign" in str(sub.get("organizationName", "")) for sub in subjects)
    assert context.check_hostname and context.verify_mode.name == "CERT_REQUIRED"
    monkeypatch.setenv("SSL_CERT_FILE", str(bundle))
    CL.make_client(CFG, tmp_path / "other")
    assert not (tmp_path / "other" / "tls-ca-bundle.pem").exists()


class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)
        self.finish_reason = "stop"


class _Usage:
    def model_dump(self):
        return {"completion_tokens": 1}


class _Response:
    def __init__(self, content):
        self.choices = [_Choice(content)]
        self.usage = _Usage()

    def model_dump(self):
        return {"content": self.choices[0].message.content}


class _Model:
    def __init__(self, id):
        self.id = id


class FakeClient:
    """Serves some models; answers with the reference program; can lose a model."""

    def __init__(self, served, answer_for, gone_after=None, list_error=None):
        self.served = served
        self.answer_for = answer_for
        self.gone_after = gone_after or {}
        self.calls = {}
        self.list_error = list_error
        client = self

        class _Models:
            def list(self):
                if client.list_error:
                    raise client.list_error
                return type("L", (), {"data": [_Model(m) for m in client.served]})()

        class _Completions:
            def create(self, **kw):
                import openai

                model = kw["model"]
                client.calls[model] = client.calls.get(model, 0) + 1
                limit = client.gone_after.get(model)
                if limit is not None and client.calls[model] > limit:
                    raise openai.NotFoundError(
                        message="Model 'x' not found (cached)",
                        response=type(
                            "R",
                            (),
                            {"status_code": 404, "headers": {}, "request": None},
                        )(),
                        body={"type": "model_not_found"},
                    )
                return _Response(client.answer_for(kw))

        self.models = _Models()
        self.chat = type("Chat", (), {"completions": _Completions()})()


def _answer_from_messages(prepared):
    """Map a prompt back to its copy and answer with the reference program."""
    index = {}
    for problem, copies in prepared:
        for copy in copies:
            text = C.prompts.user_text(copy.problem.given, copy.problem.description)
            index[text] = _golden(copy)

    def answer(kw):
        return index[kw["messages"][1]["content"]]

    return answer


def test_run_skips_unserved_models_and_reports_exit_codes(tmp_path, monkeypatch):
    cfg = _cfg(models=("fake/served", "fake/absent"))
    out = tmp_path / "e0"
    prepared = C.prepare(cfg, out)
    client = FakeClient({"fake/served", "other"}, _answer_from_messages(prepared))

    class Creds:
        source = "test"
        expires_at = None

    monkeypatch.setattr(C, "make_client", lambda cfg, ca_dir=None: (client, Creds()))
    code = C.run(cfg, out, workers=2)
    assert code == C.EXIT_MODELS_ABSENT
    manifest = C.read_json(out / "run.json")
    assert (
        manifest["skipped"] == ["fake/absent"] and "fake/served" in manifest["served"]
    )
    assert client.calls == {"fake/served": 6 * 4}
    assert not (out / "fake_absent").exists()
    payload, text = C.summarize(cfg, out)
    assert payload["unevaluated_models"] == ["fake/absent"]
    assert "No evaluations for: fake/absent" in text
    assert payload["gate"]["models"]["fake/absent"]["incomplete"]
    # the served model is perfectly consistent, which is decisive on its own
    assert payload["gate"]["verdict"] == "H0 FALSIFIED"
    # a second run has nothing left to do for the served model and still exits 5
    assert C.run(cfg, out, workers=2) == C.EXIT_MODELS_ABSENT
    assert client.calls == {"fake/served": 6 * 4}


def test_run_aborts_when_the_listing_fails(tmp_path, monkeypatch):
    import openai

    cfg = _cfg()
    out = tmp_path / "e0"
    C.prepare(cfg, out)
    err = openai.APIConnectionError(request=None)  # type: ignore[arg-type]
    client = FakeClient({"fake/model"}, lambda kw: "", list_error=err)

    class Creds:
        source = "test"
        expires_at = None

    monkeypatch.setattr(C, "make_client", lambda cfg, ca_dir=None: (client, Creds()))
    assert C.run(cfg, out) == C.EXIT_ENDPOINT_UNAVAILABLE
    assert client.calls == {}
    assert "could not list models" in C.read_json(out / "run.json")["error"]
    assert not C.lock_is_held(out)


def test_model_unloaded_mid_run_stops_its_remaining_copies(tmp_path, monkeypatch):
    cfg = _cfg(models=("fake/flaky",), copies=4, per_bucket=2)
    out = tmp_path / "e0"
    prepared = C.prepare(cfg, out)
    client = FakeClient(
        {"fake/flaky"}, _answer_from_messages(prepared), gone_after={"fake/flaky": 5}
    )

    class Creds:
        source = "test"
        expires_at = None

    monkeypatch.setattr(C, "make_client", lambda cfg, ca_dir=None: (client, Creds()))
    code = C.run(cfg, out, workers=1)
    assert code == C.EXIT_FAILED_REQUESTS  # the one cached 404
    manifest = C.read_json(out / "run.json")
    assert (
        manifest["gone_mid_run"] == ["fake/flaky"] and manifest["failed_requests"] == 1
    )
    evals = list(out.glob("fake_flaky/*/c*/eval.json"))
    results = list(out.glob("fake_flaky/*/c*/result.json"))
    assert (
        len(evals) == 5 and len(results) == 6
    )  # 5 answers, 1 cached 404, 18 untouched
    assert client.calls == {"fake/flaky": 6}
    archived = C.recover_errors(out)
    assert len(archived) == 1
