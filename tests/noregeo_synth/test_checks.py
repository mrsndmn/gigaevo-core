"""The acceptance checks, which are the only thing standing between a bad
generation and a corrupted dataset.

Every failure guarded here is silent by construction: a step that draws nothing
still produces a PNG, a step that erases the triangle still "changed" the
picture, a step that resizes the figure makes consecutive frames incomparable,
and a solution that retypes the golden answer still matches the golden answer.
None of these raise; they just quietly ship a sample whose reasoning and image
say different things.

No network and no inference: the model's part is hand-written here.
"""

from __future__ import annotations

import json

import pytest

from experiments.noregeo_synth import dataset as ds
from experiments.noregeo_synth import guard
from experiments.noregeo_synth.canvas import build_preamble, view_box
from experiments.noregeo_synth.configs import CONFIGS_BY_NAME
from experiments.noregeo_synth.execute import run_steps
from experiments.noregeo_synth.generate import (
    RecordStore,
    _add_usage,
    _call,
    _seed_for,
    _stratified,
    accepted_task_ids,
    failure_record,
    golden_point,
    run_checks,
    upsert_record,
)
from experiments.noregeo_synth.verify import answer_matches, literal_leak

CFG = CONFIGS_BY_NAME["qwen38-27b"]

POINT_ROW = {
    "qid": 6200,
    "schema_type": "point",
    "question": "Triangle A=(7, 2), B=(-6, -5), C=(-6, -9). Find the circumcenter.",
    "points": [[7, 2], [-6, -5], [-6, -9]],
    "answer": {"point": [3.4615384615384617, -7.0]},
    "choices": None,
}

DRAW_TRIANGLE = (
    "tri = np.array([P['A'], P['B'], P['C'], P['A']])\n"
    "ax.plot(tri[:, 0], tri[:, 1], color='tab:blue', lw=2)\n"
    "mark('A', P['A'])\n"
)
DRAW_ANSWER = (
    "mid_bc = (P['B'] + P['C']) / 2\n"
    "mid_ab = (P['A'] + P['B']) / 2\n"
    "d = P['B'] - P['A']\n"
    "n = np.array([d[1], -d[0]], dtype=float)\n"
    "O = mid_ab + (mid_bc[1] - mid_ab[1]) / n[1] * n\n"
    "mark('O', O, color='red')\n"
    "state['answer'] = O\n"
)


def _solution(*codes: str) -> dict:
    return {
        "plan": "test",
        "steps": [
            {"title": f"step {i}", "reasoning": "because", "code": c}
            for i, c in enumerate(codes, 1)
        ],
        "final_answer": {"x": 3.4615384615384617, "y": -7.0},
    }


def _check(tmp_path, *codes: str, row: dict = POINT_ROW, cfg=CFG):
    solution = _solution(*codes)
    exec_res = run_steps(
        build_preamble(row, cfg), solution["steps"], tmp_path, cfg, golden_point(row)
    )
    return exec_res, run_checks(row, solution, exec_res, cfg)


# ---------------------------------------------------------------------------
# Incrementality
# ---------------------------------------------------------------------------


def test_good_solution_is_accepted(tmp_path):
    exec_res, checks = _check(tmp_path, DRAW_TRIANGLE, DRAW_ANSWER)
    assert exec_res.ok, exec_res.error
    assert checks["problems"] == []
    assert checks["incremental_ok"] and checks["drawn_ok"] and checks["code_answer_ok"]


def test_first_step_that_draws_nothing_is_rejected(tmp_path):
    """Compared against a blank canvas, `pass` would "change" ~3% of pixels:
    the grid and the axes that the preamble already drew."""
    _, checks = _check(tmp_path, "pass", DRAW_TRIANGLE, DRAW_ANSWER)
    assert checks["incremental_ok"] is False
    assert any("drew nothing new" in p for p in checks["problems"])
    assert checks["deltas"][0] == pytest.approx(0.0, abs=1e-6)


def test_later_step_that_draws_nothing_is_rejected(tmp_path):
    _, checks = _check(tmp_path, DRAW_TRIANGLE, "x = 1 + 1", DRAW_ANSWER)
    assert checks["incremental_ok"] is False
    assert any("drew nothing new" in p for p in checks["problems"])


def test_step_that_erases_earlier_content_is_rejected(tmp_path):
    """A step may add. Painting over the triangle changes pixels — which is
    what a naive "did something change?" check rewards."""
    erase = (
        "ax.add_patch(plt.Rectangle((-20, -20), 60, 60, color='white', zorder=9))\n"
        "mark('X', np.array([0.0, 0.0]), color='black')\n"
    )
    _, checks = _check(tmp_path, DRAW_TRIANGLE, erase, DRAW_ANSWER)
    assert checks["incremental_ok"] is False
    assert any("erased" in p for p in checks["problems"])


def test_a_legend_that_moves_is_not_erasure(tmp_path):
    """Seen in a real generation: the last step added a labelled artist, the
    `loc="best"` legend jumped corners, and 7% of the ink "vanished" with it."""
    with_legend = DRAW_TRIANGLE + (
        "ax.plot([-6, 7], [-9, 2], color='tab:purple', label='AC')\n"
        "ax.legend(loc='upper right', fontsize=8)\n"
    )
    moved = DRAW_ANSWER + "ax.legend(loc='lower left', fontsize=8)\n"
    _, checks = _check(tmp_path, with_legend, moved)
    assert checks["problems"] == []
    assert checks["incremental_ok"] is True


def test_ink_loss_is_not_reported_when_the_view_grows(tmp_path):
    """expand_view moves every pixel; that is not erasure."""
    grow = "expand_view(np.array([40.0, 40.0]), margin=2.0)\nmark('Z', P['C'])\n"
    _, checks = _check(tmp_path, DRAW_TRIANGLE, grow, DRAW_ANSWER)
    assert checks["ink_losses"][1] is None
    assert not any("erased" in p for p in checks["problems"])


# ---------------------------------------------------------------------------
# Answer checks
# ---------------------------------------------------------------------------


def test_missing_state_answer_is_rejected(tmp_path):
    no_state = DRAW_ANSWER.replace("state['answer'] = O\n", "")
    _, checks = _check(tmp_path, DRAW_TRIANGLE, no_state)
    assert checks["code_answer_ok"] is False
    assert any("state[" in p for p in checks["problems"])


def test_code_that_computes_a_different_point_is_rejected(tmp_path):
    """The regression this whole check exists for: right prose, wrong drawing.

    A sign flip in the perpendicular direction puts the circumcenter at
    (-2.46, 4.0) while `final_answer` still carries the golden value.
    """
    wrong = (
        "mid_ab = (P['A'] + P['B']) / 2\n"
        "d = P['B'] - P['A']\n"
        "n = np.array([d[1], -d[0]], dtype=float)\n"
        "O = mid_ab + (5.5 / 13.0) * n\n"
        "mark('O', O, color='red')\n"
        "state['answer'] = O\n"
    )
    _, checks = _check(tmp_path, DRAW_TRIANGLE, wrong)
    assert checks["code_answer_ok"] is False
    assert checks["drawn_ok"] is False
    assert checks["answer_ok"] is True  # the prose alone would have passed


def test_unchecked_signals_are_none_not_true(tmp_path):
    """A sample that never ran must not report `drawn_ok=True`."""
    _, checks = _check(tmp_path, "raise ValueError('boom')")
    assert checks["code_answer_ok"] is None
    assert checks["drawn_ok"] is None
    assert checks["answer_ok"] is None
    assert checks["incremental_ok"] is None
    assert checks["problems"]


def test_answer_matches_per_schema():
    cfg = CFG
    cls_row = {"schema_type": "classification", "answer": "DEF"}
    assert answer_matches(cls_row, "DEF", cfg)[0]
    assert answer_matches(cls_row, "def", cfg)[0]
    assert not answer_matches(cls_row, "ABC", cfg)[0]

    num_row = {"schema_type": "number", "answer": 17.5}
    assert answer_matches(num_row, 17.5, cfg)[0]
    assert answer_matches(num_row, 17.504, cfg)[0]
    assert not answer_matches(num_row, 18.0, cfg)[0]
    assert not answer_matches(num_row, "nope", cfg)[0]

    assert answer_matches(POINT_ROW, {"x": 3.4615384615, "y": -7.0}, cfg)[0]
    assert not answer_matches(POINT_ROW, {"x": 3.0, "y": -7.0}, cfg)[0]
    assert not answer_matches(POINT_ROW, {"x": None, "y": -7.0}, cfg)[0]


def test_literal_leak_flags_a_pasted_answer_but_not_given_coordinates():
    pasted = [{"code": "O = np.array([3.4615384615384617, -7.0])"}]
    assert literal_leak(POINT_ROW, pasted)
    # one matching coordinate is not enough to accuse a step
    assert not literal_leak(POINT_ROW, [{"code": "y = -7.0"}])
    # numbers that come from the problem statement are fair game
    assert not literal_leak(POINT_ROW, [{"code": "A = np.array([7, 2])"}])
    num_row = {"schema_type": "number", "answer": 17.5, "question": "A=(-1, 0)."}
    assert literal_leak(num_row, [{"code": "area = 17.5"}])
    assert not literal_leak(num_row, [{"code": "area = 0.5 * b * h"}])


# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "code",
    [
        "ax.clear()",
        "ax.cla()",
        "plt.close(fig)",
        "fig.savefig('/tmp/x.png')",
        "fig.set_size_inches(7, 7)",  # different PNG size => uncomparable frames
        "ax.set_xlim(0, 1)",  # zooming in hides earlier content
        "del ax.lines[0]",
        "getattr(ax, 'cla')()",
        "ax.lines[0].set_visible(False)",
        "import os",
        "from subprocess import run",
        "np.save('/tmp/x.npy', P['A'])",
        "open('/tmp/x', 'w').write('hi')",
        "__builtins__['open']('/tmp/x', 'w')",
        "ax.plot([0, 1], [0, 1]",  # syntax error
    ],
)
def test_guard_blocks(code):
    assert guard.check(code), f"guard let through: {code}"


@pytest.mark.parametrize("code", [DRAW_TRIANGLE, DRAW_ANSWER, "expand_view(P['A'])"])
def test_guard_allows_normal_plotting(code):
    assert guard.check(code) == []


def test_guard_rejection_leaves_no_orphan_images(tmp_path):
    """A blocked repair round must not leave the previous round's PNGs behind
    pretending to belong to the record."""
    exec_ok, _ = _check(tmp_path, DRAW_TRIANGLE, DRAW_ANSWER)
    assert exec_ok.ok and list(tmp_path.glob("step_*.png"))
    blocked = run_steps(
        build_preamble(POINT_ROW, CFG),
        [{"title": "t", "code": "ax.clear()"}],
        tmp_path,
        CFG,
    )
    assert not blocked.ok
    assert list(tmp_path.glob("step_*.png")) == []


# ---------------------------------------------------------------------------
# Dataset / canvas plumbing
# ---------------------------------------------------------------------------


def test_named_points_handles_every_shape_in_the_benchmark():
    assert ds.named_points({"points": None}) == {}
    assert list(ds.named_points({"points": [[1, 2], [3, 4]]})) == ["A", "B"]
    assert list(ds.named_points({"points": {"P": [1, 2]}})) == ["P"]
    poly = ds.named_points({"points": {"polygon": [[0, 0], [1, 0], [1, 1]]}})
    assert poly["polygon"] == [[0, 0], [1, 0], [1, 1]]


def test_view_box_includes_the_answer_point():
    x0, x1, y0, y1 = view_box(POINT_ROW)
    ax_, ay = POINT_ROW["answer"]["point"]
    assert x0 < ax_ < x1 and y0 < ay < y1


def test_preamble_binds_points_and_helpers():
    src = build_preamble(POINT_ROW, CFG)
    assert '"A": np.array([7' in src
    assert "def expand_view" in src and "def mark" in src


def test_select_rejects_ambiguous_and_malformed_selectors():
    tasks = [
        ds.Task("alpha", "a.jsonl", {"qid": 10}),
        ds.Task("beta", "b.jsonl", {"qid": 10}),
    ]
    assert ds.select(tasks, "alpha:10").stem == "alpha"
    with pytest.raises(KeyError, match="ambiguous"):
        ds.select(tasks, "10")
    with pytest.raises(KeyError, match="integer qid"):
        ds.select(tasks, "alpha:x")
    with pytest.raises(KeyError, match="no NoReGeo task"):
        ds.select(tasks, "alpha:11")


# ---------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------


def test_upsert_record_replaces_instead_of_appending(tmp_path):
    """The sample directory is rewritten on every run; an append-only log would
    leave old rows pointing at images that belong to a newer run."""
    path = tmp_path / "records.jsonl"
    upsert_record(path, {"task_id": "a:1", "config": "c", "steps": [1]})
    upsert_record(path, {"task_id": "b:2", "config": "c", "steps": [1]})
    upsert_record(path, {"task_id": "a:1", "config": "c", "steps": [1, 2]})
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["task_id"] for r in rows] == ["a:1", "b:2"]
    assert rows[0]["steps"] == [1, 2]
    # a different arm keeps its own row for the same task
    upsert_record(path, {"task_id": "a:1", "config": "other", "steps": []})
    assert len(path.read_text().splitlines()) == 3


def test_record_store_batches_writes_and_survives_close(tmp_path):
    path = tmp_path / "records.jsonl"
    store = RecordStore(path, flush_interval=3600)  # never flush on its own
    store.add({"task_id": "a:1", "config": "c", "steps": []})
    assert not path.exists()  # batched, not written per record
    store.close()
    assert len(path.read_text().splitlines()) == 1
    # a second store picks the existing rows up instead of truncating them
    RecordStore(path, flush_interval=0).add({"task_id": "b:2", "config": "c"})
    assert len(path.read_text().splitlines()) == 2


# ---------------------------------------------------------------------------
# Batch-run bookkeeping
# ---------------------------------------------------------------------------


def _fake_task(stem="alpha", qid=1, category="Cat A"):
    return ds.Task(stem, f"{stem}.jsonl", {"qid": qid, "category": category})


def test_resume_ignores_rows_whose_images_are_gone(tmp_path):
    """A run killed mid-sample wipes the directory but leaves the old accepted
    row behind; resuming must regenerate it, not skip it forever."""
    path = tmp_path / "records.jsonl"
    sample = tmp_path / "alpha_1"
    sample.mkdir()
    (sample / "record.json").write_text("{}")
    (sample / "step_01.png").write_bytes(b"x")
    row = {
        "task_id": "alpha:1",
        "config": "c",
        "accepted": True,
        "steps": [{"image": "step_01.png"}],
        "sample_dir": str(sample),
    }
    upsert_record(path, row)
    assert accepted_task_ids(path) == {"alpha:1"}

    (sample / "step_01.png").unlink()  # the interrupted rerun wiped it
    assert accepted_task_ids(path) == set()


def test_failure_record_is_written_for_a_sample_that_never_ran(tmp_path):
    rec = failure_record(_fake_task(), CFG, tmp_path, RuntimeError("boom"))
    assert rec["accepted"] is False
    assert rec["steps"] == []
    assert "boom" in rec["problems"][0]
    # and it is enough to keep --resume honest
    path = tmp_path / "records.jsonl"
    upsert_record(path, rec)
    assert accepted_task_ids(path) == set()


def test_stratified_takes_from_every_group_and_is_deterministic():
    tasks = [_fake_task("alpha", q, "Cat A") for q in range(10)] + [
        _fake_task("beta", q, "Cat B") for q in range(10)
    ]
    picked = _stratified(tasks, lambda t: t.stem, 3, seed=0)
    assert len(picked) == 6
    assert {t.stem for t in picked} == {"alpha", "beta"}
    assert [t.task_id for t in picked] == [
        t.task_id for t in _stratified(tasks, lambda t: t.stem, 3, seed=0)
    ]
    # asking for more than a group holds takes the whole group, not an error
    assert len(_stratified(tasks, lambda t: t.stem, 50, seed=0)) == 20


def test_seed_is_per_task_so_a_resume_does_not_replay_the_same_roll():
    a0 = _seed_for(CFG, "alpha:1", 0)
    b0 = _seed_for(CFG, "beta:2", 0)
    a1 = _seed_for(CFG, "alpha:1", 1)
    assert a0 != b0, "different tasks must not share one seed"
    assert a0 != a1, "a retry must re-roll"
    assert a0 == _seed_for(CFG, "alpha:1", 0), "same task must be reproducible"


class _FakeChoice:
    def __init__(self, content, finish_reason="stop"):
        self.message = type("M", (), {"content": content, "reasoning": None})()
        self.finish_reason = finish_reason


class _FakeClient:
    """Returns the queued responses in order, recording the request kwargs."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                content, finish = outer.responses.pop(0)
                return type(
                    "R",
                    (),
                    {
                        "choices": [_FakeChoice(content, finish)],
                        "usage": type(
                            "U",
                            (),
                            {"model_dump": lambda self: {"completion_tokens": 100}},
                        )(),
                    },
                )()

        self.chat = type("C", (), {"completions": _Completions()})()


def test_call_retry_escalates_the_budget_and_bills_failed_attempts():
    """Truncation has two causes: a degenerate sampler (re-roll) and a solution
    that did not fit (raise the cap). The retry has to try both."""
    client = _FakeClient(
        [
            ("{'broken", "length"),
            ("{'broken", "length"),
            ('{"ok": 1}', "stop"),
        ]
    )
    solution, _, usage = _call(client, CFG, [{"role": "user", "content": "x"}], {}, "t")
    assert solution == {"ok": 1}
    budgets = [c["max_tokens"] for c in client.calls]
    assert budgets == [CFG.max_tokens, CFG.max_tokens, CFG.max_tokens * 2]
    seeds = [c["seed"] for c in client.calls]
    assert len(set(seeds)) == 3, "every attempt must use a fresh seed"
    assert usage["completion_tokens"] == 300, "failed attempts cost tokens too"


def test_call_retries_a_transport_failure(monkeypatch):
    """A gateway hiccup makes the SDK raise before we ever see a message; the
    full 2700-sample run lost a sample to exactly this."""
    monkeypatch.setattr("time.sleep", lambda *_: None)
    client = _FakeClient([('{"ok": 1}', "stop")])
    original = client.chat.completions.create

    def flaky(**kwargs):
        if not client.calls:
            client.calls.append(kwargs)
            raise json.JSONDecodeError("Expecting value", "", 0)
        return original(**kwargs)

    client.chat.completions.create = flaky
    solution, _, _ = _call(client, CFG, [{"role": "user", "content": "x"}], {}, "t")
    assert solution == {"ok": 1}
    assert len(client.calls) == 2


def test_call_gives_up_and_raises_after_the_configured_retries():
    client = _FakeClient([("{'broken", "length")] * (CFG.call_retries + 1))
    with pytest.raises(RuntimeError, match="no usable completion"):
        _call(client, CFG, [{"role": "user", "content": "x"}], {}, "t")
    assert len(client.calls) == CFG.call_retries + 1


def test_records_are_strict_json(tmp_path):
    """NaN is valid for `json.dump` and invalid for jq, serde and pyarrow."""
    _, checks = _check(tmp_path, DRAW_TRIANGLE, DRAW_ANSWER)
    text = json.dumps(checks)
    json.loads(text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))


def test_add_usage_sums_nested_counters():
    total = {"completion_tokens": 5, "details": {"reasoning_tokens": 5}}
    total = _add_usage(
        total, {"completion_tokens": 7, "details": {"reasoning_tokens": 6}}
    )
    assert total["completion_tokens"] == 12
    assert total["details"]["reasoning_tokens"] == 11
