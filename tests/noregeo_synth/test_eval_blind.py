"""The coordinate-blind eval, whose failure modes are all silent.

A prompt that still contains the coordinates turns the vision arm back into
arithmetic and scores *well*. A figure taken from `images_full/` carries the
answer in its title and also scores well. A wrong scale rule in the
falsification arm marks a model that read the picture correctly as wrong. None
of those raise; they produce a plausible number that means the opposite of what
it looks like.

No network and no inference: every model answer here is hand-written.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from experiments.noregeo_synth import dataset as ds
from experiments.noregeo_synth import eval_blind as eb
from experiments.noregeo_synth.configs import (
    ARM_BOTH,
    ARM_FIGURE,
    ARM_SWAP,
    ARM_SWAP_CONTROL,
    ARM_TEXT,
    ARMS,
    EVAL_CONFIGS_BY_NAME,
    EXCLUDED_STEMS,
    FIGURE_ANSWER_IN_TITLE,
    SWAP_RULE,
    SWAP_SCALE,
)
from experiments.noregeo_synth.figures import (
    DEFAULT_VIEW,
    iter_points,
    render_dots,
    scale_points,
    union_view_box,
    view_box_for,
)

CFG = EVAL_CONFIGS_BY_NAME["qwen38-27b"]

TRIANGLE_AREA = {
    "qid": 9000,
    "schema_type": "number",
    "category": "Basic coordinate tasks",
    "difficulty": 2,
    "points": {"A": [-1, 0], "B": [2, -7], "C": [4, 0]},
    "answer": 17.5,
    "question": "Triangle A=(-1, 0), B=(2, -7), C=(4, 0). What is its area?",
    "image_question": "Refer to the image. What is the area of the triangle ABC?",
    "image_answer": 17.5,
    "choices": None,
    "image_choices": None,
    "im_path": "/images_full/task_09000_triangle_area.png",
    "im_dots_path": "/images/task_09000_triangle_area.png",
}

CIRCUMCENTER = {
    "qid": 6200,
    "schema_type": "point",
    "category": "Simple circle properties",
    "difficulty": 2,
    "points": [[7, 2], [-6, -5], [-6, -9]],
    "answer": {"point": [3.4615384615384617, -7.0]},
    "question": "Triangle A=(7, 2), B=(-6, -5), C=(-6, -9). Find the circumcenter.",
    "image_question": "Refer to the input image. What is the circumcenter?",
    "image_answer": {"point": [3.4615384615384617, -7.0]},
    "choices": None,
    "image_choices": None,
    "im_path": "/images_full/task_06200_circumcircle.png",
    "im_dots_path": "/images/task_06200_circumcircle.png",
}

SHAPE_SYMMETRY = {
    "qid": 13100,
    "schema_type": "classification",
    "category": "Symmetry",
    "difficulty": 2,
    "points": {"polygon": [[-4, 1], [-4, -7], [-6, -2], [-9, 5], [2, 10]]},
    "answer": "no",
    "question": (
        "Polygon vertices [[-4, 1], [-4, -7], [-6, -2], [-9, 5], [2, 10]], "
        "symmetric about X-axis?"
    ),
    "image_question": "Refer to the image. Is the polygon symmetric about the {axis}-axis?",
    "image_answer": "no",
    "choices": ["yes", "no"],
    "image_choices": ["yes", "no"],
    "im_path": "/images_full/task_13100_shape_symmetry.png",
    "im_dots_path": "/images/task_13100_shape_symmetry.png",
}

SYMMETRY = {
    "qid": 13000,
    "schema_type": "classification",
    "category": "Symmetry",
    "difficulty": 2,
    "points": {"P": [0, 1], "Q": [-8, -7]},
    "answer": "yes",
    "question": "Points P=(0, 1), Q=(-8, -7) symmetric about point O=(-4, -3)?",
    "image_question": "Refer to the image. Are P and Q symmetric about None?",
    "image_answer": "yes",
    "choices": ["yes", "no"],
    "image_choices": ["yes", "no"],
    "im_path": "/images_full/task_13000_symmetry.png",
    "im_dots_path": "/images/task_13000_symmetry.png",
}


def task(row: dict, stem: str) -> ds.Task:
    return ds.Task(stem=stem, file=f"x/{stem}_tasks.jsonl", row=row)


@pytest.fixture
def no_download(monkeypatch, tmp_path):
    """`fetch_image` without the network: a real PNG at the expected path."""
    written: list[str] = []

    def fake_fetch(repo_path: str) -> Path:
        written.append(repo_path)
        dest = tmp_path / repo_path.lstrip("/")
        render_dots({"A": [0, 0]}, "stub", dest)
        return dest

    monkeypatch.setattr(ds, "fetch_image", fake_fetch)
    monkeypatch.setattr(eb.ds, "fetch_image", fake_fetch)
    return written


def prompt_text(request: dict) -> str:
    content = request["messages"][1]["content"]
    return content if isinstance(content, str) else content[0]["text"]


# ---------------------------------------------------------------------------
# The prompt must not contain what the model is supposed to read off the figure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "row,stem",
    [(TRIANGLE_AREA, "triangle_area"), (CIRCUMCENTER, "outer_circle_center")],
)
def test_figure_arm_prompt_carries_no_coordinates(row, stem, no_download, tmp_path):
    req = eb.build_request(task(row, stem), ARM_FIGURE, CFG, tmp_path)
    text = prompt_text(req)
    assert req["meta"]["leaked_coordinates"] == []
    for _, (x, y) in iter_points(ds.named_points(row)):
        assert f"{x:g}, {y:g}" not in text
        assert f"({x:g},{y:g})" not in text


def test_leak_detector_actually_fires():
    points = ds.named_points(TRIANGLE_AREA)
    assert eb.coordinate_leak("nothing here", points) == []
    assert eb.coordinate_leak("the point (2, -7) is given", points) == ["(2, -7)"]
    # A pair that is not part of the configuration is not a leak.
    assert eb.coordinate_leak("(11, 11)", points) == []


def test_text_arm_does_carry_coordinates(tmp_path):
    """The control for the test above: the text arm is supposed to leak."""
    req = eb.build_request(
        task(TRIANGLE_AREA, "triangle_area"), ARM_TEXT, CFG, tmp_path
    )
    assert "(-1, 0)" in prompt_text(req)
    assert req["messages"][1]["content"] == prompt_text(req)  # no image attached


# ---------------------------------------------------------------------------
# The figure must be the one that does NOT print the answer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arm", [ARM_FIGURE, ARM_BOTH])
def test_only_the_dots_image_is_ever_fetched(arm, no_download, tmp_path):
    """`images_full/` renders the answer into the title ("area = 17.50")."""
    eb.build_request(task(TRIANGLE_AREA, "triangle_area"), arm, CFG, tmp_path)
    assert no_download == [TRIANGLE_AREA["im_dots_path"]]
    assert all("images_full" not in p for p in no_download)


def test_figure_arms_attach_exactly_one_image(no_download, tmp_path):
    req = eb.build_request(
        task(TRIANGLE_AREA, "triangle_area"), ARM_FIGURE, CFG, tmp_path
    )
    content = req["messages"][1]["content"]
    images = [c for c in content if c["type"] == "image_url"]
    assert len(images) == 1
    assert images[0]["image_url"]["url"].startswith("data:image/png;base64,")


# ---------------------------------------------------------------------------
# Question repair
# ---------------------------------------------------------------------------


def test_repair_fills_the_axis_placeholder():
    question, meta = eb.repair_image_question(task(SHAPE_SYMMETRY, "shape_symmetry"))
    assert "{axis}" not in question
    assert "symmetric about the X-axis" in question
    assert meta["repaired"] == "X"
    assert meta["coords_injected"] is False
    # ...but the figure still cannot carry this question, and says so.
    assert meta["figure_underspecified"] is True


def test_repair_of_the_symmetry_centre_is_flagged_as_injecting_coordinates():
    question, meta = eb.repair_image_question(task(SYMMETRY, "symmetry"))
    assert "None" not in question
    assert "point O=(-4, -3)" in question
    assert meta["coords_injected"] is True


def test_repair_is_a_no_op_when_the_template_was_rendered():
    row = {
        **SYMMETRY,
        "image_question": "Refer to the image. Are P and Q symmetric about X?",
    }
    question, meta = eb.repair_image_question(task(row, "symmetry"))
    assert question == row["image_question"]
    assert meta["repaired"] is None and meta["coords_injected"] is False


def test_unrepairable_row_is_marked_not_silently_passed():
    row = {**SHAPE_SYMMETRY, "question": "Polygon vertices [[0, 0]], symmetric?"}
    _, meta = eb.repair_image_question(task(row, "shape_symmetry"))
    assert meta["repaired"] == "FAILED"


# ---------------------------------------------------------------------------
# The falsification arm: the scale rules have to match the real geometry
# ---------------------------------------------------------------------------


def _shoelace(pts):
    (x1, y1), (x2, y2), (x3, y3) = pts
    return abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2


def _circumcenter(pts):
    (ax, ay), (bx, by), (cx, cy) = pts
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    ux = (
        (ax**2 + ay**2) * (by - cy)
        + (bx**2 + by**2) * (cy - ay)
        + (cx**2 + cy**2) * (ay - by)
    ) / d
    uy = (
        (ax**2 + ay**2) * (cx - bx)
        + (bx**2 + by**2) * (ax - cx)
        + (cx**2 + cy**2) * (bx - ax)
    ) / d
    return [ux, uy]


def test_area_rule_matches_an_independent_solver():
    pts = [v for _, v in iter_points(ds.named_points(TRIANGLE_AREA))]
    scaled = [
        v
        for _, v in iter_points(
            scale_points(ds.named_points(TRIANGLE_AREA), SWAP_SCALE)
        )
    ]
    assert _shoelace(pts) == pytest.approx(TRIANGLE_AREA["answer"])
    claimed = eb.swapped_answer(task(TRIANGLE_AREA, "triangle_area"), SWAP_SCALE)
    assert claimed == pytest.approx(_shoelace(scaled))


def test_point_rule_matches_an_independent_solver():
    pts = [v for _, v in iter_points(ds.named_points(CIRCUMCENTER))]
    scaled = [
        v
        for _, v in iter_points(scale_points(ds.named_points(CIRCUMCENTER), SWAP_SCALE))
    ]
    assert _circumcenter(pts) == pytest.approx(CIRCUMCENTER["answer"]["point"])
    claimed = eb.swapped_answer(task(CIRCUMCENTER, "outer_circle_center"), SWAP_SCALE)
    assert claimed["point"] == pytest.approx(_circumcenter(scaled))


def test_length_rule_matches_an_independent_solver():
    row = {
        "qid": 1,
        "schema_type": "number",
        "points": {"A": [-9, -4], "B": [-5, 7]},
        "answer": math.dist([-9, -4], [-5, 7]),
        "question": "Points A=(-9, -4), B=(-5, 7). What is the length of segment AB?",
        "choices": None,
    }
    claimed = eb.swapped_answer(task(row, "segment_length"), SWAP_SCALE)
    assert claimed == pytest.approx(
        math.dist([-9 * SWAP_SCALE, -4 * SWAP_SCALE], [-5 * SWAP_SCALE, 7 * SWAP_SCALE])
    )


def test_every_swap_stem_has_a_rule_and_no_classification_stem_does():
    assert set(SWAP_RULE.values()) <= {"area", "length", "point"}
    for stem in (
        "collinearity",
        "parallel_lines",
        "perpendicular",
        "right_angle",
        "shape_symmetry",
        "symmetry",
        "special_lines",
        "semicircle_triangle",
        "circle_diameter",
        "triangle_size",
        "parallelogram_size",
    ):
        assert stem not in SWAP_RULE, (
            f"{stem} is scale-invariant, it cannot discriminate"
        )


@pytest.mark.parametrize("arm", [ARM_SWAP, ARM_SWAP_CONTROL])
def test_classification_tasks_are_skipped_by_the_swap_arms(arm):
    assert eb.arm_applies(task(SHAPE_SYMMETRY, "shape_symmetry"), arm) is not None
    assert eb.arm_applies(task(TRIANGLE_AREA, "triangle_area"), arm) is None


def test_swap_and_control_share_one_view_box(tmp_path):
    """Otherwise the perturbed figure differs only in its tick labels."""
    t = task(CIRCUMCENTER, "outer_circle_center")
    swap = eb.build_request(t, ARM_SWAP, CFG, tmp_path)
    control = eb.build_request(t, ARM_SWAP_CONTROL, CFG, tmp_path)
    assert swap["meta"]["scale"] == SWAP_SCALE
    assert control["meta"]["scale"] == 1.0
    # The golden answer moves with the figure...
    assert swap["golden"]["point"][0] == pytest.approx(
        CIRCUMCENTER["answer"]["point"][0] * SWAP_SCALE
    )
    assert control["golden"] == CIRCUMCENTER["answer"]
    # ...and the text the model reads is the original, stale one in both.
    assert "(7, 2)" in prompt_text(swap)
    assert "(7, 2)" in prompt_text(control)
    assert Path(swap["meta"]["image"]).exists()
    assert Path(control["meta"]["image"]).exists()
    assert swap["meta"]["image"] != control["meta"]["image"]


def test_scaled_configuration_stays_inside_the_default_view():
    """Why the scale is 0.5 and not 2.0."""
    points = ds.named_points(CIRCUMCENTER)
    view = view_box_for(points)
    scaled = scale_points(points, SWAP_SCALE)
    for _, (x, y) in iter_points(scaled):
        assert view[0] <= x <= view[1]
        assert view[2] <= y <= view[3]


# ---------------------------------------------------------------------------
# Scoring and reporting
# ---------------------------------------------------------------------------


def test_strict_and_loose_separate_misread_from_unread():
    row = CIRCUMCENTER
    gold = row["answer"]
    near = {"x": gold["point"][0] + 0.3, "y": gold["point"][1]}
    far = {"x": 0.0, "y": 0.0}
    assert eb.score_answer(row, gold, gold["point"], CFG)["ok"] is True
    near_scored = eb.score_answer(row, gold, near, CFG)
    assert near_scored["ok"] is False and near_scored["close"] is True
    far_scored = eb.score_answer(row, gold, far, CFG)
    assert far_scored["ok"] is False and far_scored["close"] is False


def test_summary_keeps_flagged_rows_out_of_the_headline():
    rows = [
        {"arm": ARM_FIGURE, "ok": True, "close": True, "stem": "a"},
        {"arm": ARM_FIGURE, "ok": False, "close": False, "stem": "a"},
        {
            "arm": ARM_FIGURE,
            "ok": True,
            "close": True,
            "stem": "b",
            "coords_injected": True,
        },
        {
            "arm": ARM_FIGURE,
            "ok": True,
            "close": True,
            "stem": "c",
            "figure_underspecified": True,
        },
    ]
    text = eb.summarise(rows)
    line = next(ln for ln in text.splitlines() if ln.startswith(ARM_FIGURE))
    assert " 2 " in line  # two clean rows scored
    assert "50.0%" in line  # one of them right
    assert line.strip().endswith("2")  # two rows excluded and counted


def test_summary_reports_which_source_the_answer_followed():
    rows = [
        {
            "arm": ARM_SWAP,
            "ok": True,
            "close": True,
            "follows_figure": True,
            "follows_text": False,
            "stem": "a",
        },
        {
            "arm": ARM_SWAP,
            "ok": False,
            "close": False,
            "follows_figure": False,
            "follows_text": True,
            "stem": "a",
        },
        {
            "arm": ARM_SWAP,
            "ok": False,
            "close": False,
            "follows_figure": False,
            "follows_text": False,
            "stem": "a",
        },
    ]
    text = eb.summarise(rows)
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("falsification"))
    line = next(ln for ln in lines[start:] if ln.startswith(ARM_SWAP + " "))
    # one answer followed the figure, one the text, one neither
    assert line.count("33.3%") == 3


def test_records_are_keyed_per_arm(tmp_path):
    """One task evaluated on five arms must not collapse into one row."""
    from experiments.noregeo_synth.generate import RecordStore, read_records

    path = tmp_path / "records.jsonl"
    store = RecordStore(path, flush_interval=0.0)
    for arm in (ARM_TEXT, ARM_FIGURE, ARM_BOTH, ARM_SWAP, ARM_SWAP_CONTROL):
        store.add({"task_id": "triangle_area:9000", "config": f"cfg:{arm}", "arm": arm})
    store.close()
    assert len(read_records(path)) == 5


# ---------------------------------------------------------------------------
# Stems the benchmark ships broken
# ---------------------------------------------------------------------------


BISECTOR_FAR = {
    "qid": 8103,
    "schema_type": "point",
    # The real bisector:8103. Every point is far outside the benchmark's
    # default view AND far from the origin, so a box fitted to the unscaled
    # configuration excludes all three scaled points: this task rendered as an
    # empty grid before `union_view_box`.
    "points": {"A": [-13, 15], "B": [-18, -19], "C": [-13, -20]},
    "answer": {"point": [-13.0, -15.47784012887197]},
    "question": "Triangle A=(-13, 15), B=(-18, -19), C=(-13, -20); bisector at A.",
    "image_question": "Refer to the input image. Where does the bisector at A meet BC?",
    "choices": None,
    "image_choices": None,
    "im_dots_path": "/images/task_08103_bisector.png",
}

SPECIAL_LINES = {
    "qid": 10001,
    "schema_type": "classification",
    "points": {"A": [0, 0], "B": [4, 0], "C": [0, 4], "D": [2, 2]},
    "answer": "altitude",
    "question": "Triangle A=(0, 0), B=(4, 0), C=(0, 4); red segment from B to D=(2, 2). What is it?",
    "image_question": "Refer to the input image. Is the segment from B to D a median, altitude, or bisector?",
    "choices": ["median", "altitude", "bisector"],
    "image_choices": ["median", "altitude", "bisector"],
    # ...and the figure is titled "Task 10001: altitude".
    "im_dots_path": "/images/task_10001_altitude.png",
}


@pytest.mark.parametrize("stem", sorted(EXCLUDED_STEMS))
@pytest.mark.parametrize("arm", list(ARMS))
def test_broken_stems_are_skipped_on_every_arm(stem, arm):
    """outer_circle_radius asks for a length and answers with a point;
    triangle_type is a byte-identical duplicate of inner_circle_radius."""
    reason = eb.arm_applies(task({**TRIANGLE_AREA, "schema_type": "point"}, stem), arm)
    assert reason == EXCLUDED_STEMS[stem]


def test_excluded_stems_carry_no_swap_rule():
    for stem in EXCLUDED_STEMS:
        assert stem not in SWAP_RULE


@pytest.mark.parametrize("arm", [ARM_FIGURE, ARM_BOTH])
def test_stem_whose_figure_title_is_the_answer_skips_the_figure_arms(arm):
    assert eb.arm_applies(task(SPECIAL_LINES, "special_lines"), arm) is not None
    assert "title" in eb.arm_applies(task(SPECIAL_LINES, "special_lines"), arm)


def test_that_stem_still_runs_on_the_text_arm():
    assert eb.arm_applies(task(SPECIAL_LINES, "special_lines"), ARM_TEXT) is None
    assert "special_lines" in FIGURE_ANSWER_IN_TITLE


# ---------------------------------------------------------------------------
# The perturbed figure has to be visible
# ---------------------------------------------------------------------------


def test_union_view_box_holds_both_scales_when_the_default_does_not_fit():
    points = ds.named_points(BISECTOR_FAR)
    assert view_box_for(points) != DEFAULT_VIEW  # a fitted box, the risky case
    scaled = scale_points(points, SWAP_SCALE)
    fitted = view_box_for(points)
    off = [
        xy
        for _, xy in iter_points(scaled)
        if not (fitted[0] <= xy[0] <= fitted[1] and fitted[2] <= xy[1] <= fitted[3])
    ]
    assert off, "this fixture is supposed to fall outside the unscaled box"

    shared = union_view_box([points, scaled])
    for group in (points, scaled):
        for _, (x, y) in iter_points(group):
            assert shared[0] <= x <= shared[1]
            assert shared[2] <= y <= shared[3]


def test_swap_figure_is_not_blank(tmp_path):
    """The failure this guards: an empty grid scored as "did not read it"."""
    import matplotlib.image as mpimg
    import numpy as np

    t = task(BISECTOR_FAR, "bisector")
    swap = eb.build_request(t, ARM_SWAP, CFG, tmp_path)
    control = eb.build_request(t, ARM_SWAP_CONTROL, CFG, tmp_path)

    def drawn_points(path):
        arr = mpimg.imread(path)[..., :3]
        # The scatter markers are the only saturated colour on the canvas.
        return int(np.sum(np.ptp(arr, axis=-1) > 0.35))

    assert drawn_points(swap["meta"]["image"]) > 0
    assert drawn_points(control["meta"]["image"]) > 0


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def test_stem_flag_alone_does_not_sample_every_other_stem():
    from types import SimpleNamespace

    tasks = [
        task({**TRIANGLE_AREA, "qid": q}, stem)
        for stem in ("triangle_area", "bisector", "midpoint")
        for q in range(3)
    ]
    args = SimpleNamespace(stem="triangle_area", n=0, per_stem=0, task=None, seed=0)
    selected = eb._select_tasks(tasks, args)
    assert {t.stem for t in selected} == {"triangle_area"}


def test_unknown_stem_is_an_error_not_a_full_sweep():
    from types import SimpleNamespace

    tasks = [task(TRIANGLE_AREA, "triangle_area")]
    args = SimpleNamespace(stem="nope", n=0, per_stem=0, task=None, seed=0)
    with pytest.raises(SystemExit):
        eb._select_tasks(tasks, args)


# ---------------------------------------------------------------------------
# The loose columns only discriminate where the two goldens are apart
# ---------------------------------------------------------------------------


def test_loose_columns_skip_rows_where_the_two_goldens_are_indistinguishable():
    # An incircle radius of 1.3 halves to 0.65; `loose_num_atol` is 0.5, so the
    # unscaled answer passes as the scaled one and the row must not count.
    rows = [
        {
            "arm": ARM_SWAP,
            "ok": False,
            "close": True,
            "follows_figure": False,
            "follows_text": False,
            "follows_figure_close": True,
            "follows_text_close": True,
            "loose_separable": False,
            "stem": "s",
        },
        {
            "arm": ARM_SWAP,
            "ok": True,
            "close": True,
            "follows_figure": True,
            "follows_text": False,
            "follows_figure_close": True,
            "follows_text_close": False,
            "loose_separable": True,
            "stem": "s",
        },
    ]
    text = eb.summarise(rows)
    start = next(i for i, ln in enumerate(text.splitlines()) if ln.startswith("falsi"))
    line = next(ln for ln in text.splitlines()[start:] if ln.startswith(ARM_SWAP + " "))
    assert line.strip().endswith("1")  # one ambiguous row, reported not hidden
    assert "100.0%" in line  # the separable row: followed the figure


def test_figure_arm_uses_the_image_choice_list(no_download, tmp_path):
    row = {
        **SYMMETRY,
        "choices": ["yes", "no"],
        "image_choices": ["yes", "no", "maybe"],
    }
    req = eb.build_request(task(row, "symmetry"), ARM_FIGURE, CFG, tmp_path)
    enum = req["schema"]["schema"]["properties"]["final_answer"]["enum"]
    assert enum == ["yes", "no", "maybe"]
    assert "Options: ['yes', 'no', 'maybe']" in prompt_text(req)
